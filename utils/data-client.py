"""
utils/data-client.py
--------------------
Pulls a raw HSI dataset from the home-laptop server exposed by
utils/data-server.py (typically via an ngrok tunnel) down to the lab
machine.

Design goals: resumable at file granularity AND within a file, robust to
flaky home internet, and parallel enough to fill the pipe.

Usage:
    PYTHONPATH=. python utils/data-client.py \\
        --server-url https://xxxx.ngrok-free.app \\
        --dataset IIRS \\
        --workers 4

`--dest` defaults to the dataset's `raw_root` from utils.config.DATASETS,
so a plain run drops files where scripts/preprocess.sh will find them.

Resume mechanism:
  * Each file is downloaded into `<final>.partial` next to its target.
  * If `.partial` already exists from a previous crashed run, the client
    sends `Range: bytes=<local_size>-` and appends to it.
  * On success, `.partial` is atomically renamed to the final path — the
    dataloader / slice.py never see half-written files.
  * A file that already exists at the manifest's expected size is skipped.

Dependencies (see requirements.txt): requests, tqdm.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.config import DATASETS  # noqa: E402

CHUNK_SIZE = 1024 * 1024  # 1 MiB
MAX_ATTEMPTS = 8
BACKOFF_CAP_S = 30.0


def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def _download_one(
    session: requests.Session,
    server_url: str,
    dataset: str,
    entry: dict,
    dest_root: Path,
    pbar: tqdm,
    pbar_lock,
) -> tuple[str, str]:
    relpath = entry["relpath"]
    expected_size = int(entry["size"])
    final_path = dest_root / relpath
    partial_path = final_path.with_name(final_path.name + ".partial")
    final_path.parent.mkdir(parents=True, exist_ok=True)

    if final_path.exists() and final_path.stat().st_size == expected_size:
        with pbar_lock:
            pbar.update(expected_size)
        return relpath, "skipped"

    url = f"{server_url.rstrip('/')}/download/{dataset}/{relpath}"

    last_err: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        offset = 0
        if partial_path.exists():
            local_size = partial_path.stat().st_size
            if local_size > expected_size:
                partial_path.unlink()
            else:
                offset = local_size

        headers = {"Range": f"bytes={offset}-"} if offset > 0 else {}
        mode = "ab" if offset > 0 else "wb"

        # progress accounting: on this attempt, credit any bytes already on disk
        # (only the first attempt — subsequent retries will observe the same
        # offset already credited via the partial-file growth of the last try).
        credited_this_call = 0
        if attempt == 1 and offset > 0:
            with pbar_lock:
                pbar.update(offset)
                credited_this_call = offset

        try:
            with session.get(url, headers=headers, stream=True, timeout=(10, 60)) as r:
                if offset > 0 and r.status_code == 200:
                    # server ignored our Range — restart from scratch
                    partial_path.unlink(missing_ok=True)
                    with pbar_lock:
                        pbar.update(-credited_this_call)
                    raise requests.HTTPError(f"server ignored Range; got 200 for {relpath}")
                if r.status_code not in (200, 206):
                    raise requests.HTTPError(f"HTTP {r.status_code} for {relpath}: {r.text[:200]}")
                with open(partial_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        f.write(chunk)
                        with pbar_lock:
                            pbar.update(len(chunk))
            final_size = partial_path.stat().st_size
            if final_size != expected_size:
                raise IOError(
                    f"size mismatch for {relpath}: got {final_size}, expected {expected_size}"
                )
            os.replace(partial_path, final_path)
            return relpath, "downloaded"
        except Exception as e:
            last_err = e
            sleep_s = min(BACKOFF_CAP_S, 2 ** (attempt - 1))
            if attempt < MAX_ATTEMPTS:
                tqdm.write(f"[{relpath}] attempt {attempt}/{MAX_ATTEMPTS} failed: {e!r}; "
                           f"sleeping {sleep_s:.0f}s")
                time.sleep(sleep_s)
            else:
                tqdm.write(f"[{relpath}] giving up after {MAX_ATTEMPTS} attempts: {e!r}")

    return relpath, f"failed: {last_err!r}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw HSI cubes from the PRISM server.")
    parser.add_argument("--server-url", required=True, help="e.g. https://abc123.ngrok-free.app")
    parser.add_argument("--dataset", required=True, help="IIRS | M3 | AVIRIS")
    parser.add_argument("--dest", default=None,
                        help="Local root to write into. Defaults to DATASETS[<dataset>]['raw_root'].")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    key = args.dataset.upper()
    if key not in DATASETS:
        print(f"unknown dataset '{args.dataset}'. choose from {sorted(DATASETS)}", file=sys.stderr)
        sys.exit(2)

    dest_root = Path(args.dest) if args.dest else (REPO_ROOT / DATASETS[key]["raw_root"])
    dest_root.mkdir(parents=True, exist_ok=True)

    session = _make_session()

    print(f"health-checking {args.server_url} ...")
    r = session.get(f"{args.server_url.rstrip('/')}/health", timeout=15)
    r.raise_for_status()
    if not r.json().get("ok"):
        print(f"server did not report ok: {r.text}", file=sys.stderr)
        sys.exit(1)

    print(f"fetching manifest for {key} ...")
    r = session.get(f"{args.server_url.rstrip('/')}/manifest/{key}", timeout=60)
    r.raise_for_status()
    manifest: list[dict] = r.json()
    if not manifest:
        print(f"manifest for {key} is empty; nothing to download.")
        return

    total_bytes = sum(int(e["size"]) for e in manifest)
    print(f"manifest: {len(manifest)} files, {total_bytes / 1e9:.2f} GB total")
    print(f"writing to: {dest_root}")

    from threading import Lock
    pbar_lock = Lock()

    with tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024,
              desc=f"{key}", smoothing=0.05) as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(_download_one, session, args.server_url, key, entry,
                            dest_root, pbar, pbar_lock)
                for entry in manifest
            ]
            downloaded = skipped = failed = 0
            for fut in as_completed(futures):
                relpath, status = fut.result()
                if status == "downloaded":
                    downloaded += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    tqdm.write(f"FAIL {relpath}: {status}")

    print(f"done. downloaded={downloaded}, skipped={skipped}, failed={failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
