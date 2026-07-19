"""
utils/data-server.py
--------------------
Serves the raw HSI cubes (IIRS `.qub`+`.hdr`, M3 `.img`, AVIRIS `.nc`) over
HTTP so the lab machine can pull them across a home internet connection.

Meant to run behind an ngrok tunnel:

    # terminal 1 — server
    PYTHONPATH=. python utils/data-server.py
    # terminal 2 — public URL
    ngrok http 8000

Endpoints:
    GET /health                          -> {"ok": true}
    GET /datasets                        -> ["IIRS", "M3", ...] (only ones present)
    GET /manifest/{dataset}              -> [{"relpath": ..., "size": ...}, ...]
    GET /download/{dataset}/{relpath}    -> file bytes, with Range support

The Range handler is hand-rolled — Starlette's FileResponse Range behaviour
differs across versions and this way it's guaranteed to work with the
byte-resume logic in utils/data-client.py.

Dependencies (see requirements.txt): fastapi, uvicorn[standard].
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.config import DATASETS  # noqa: E402

CHUNK_SIZE = 1024 * 1024 * 10  # 10 MiB


def _dataset_root(dataset: str) -> Path:
    key = dataset.upper()
    if key not in DATASETS:
        raise HTTPException(status_code=404, detail=f"unknown dataset '{dataset}'")
    root = (REPO_ROOT / DATASETS[key]["raw_root"]).resolve()
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=404, detail=f"dataset root missing on server: {root}")
    return root


_manifest_cache: dict[str, list[dict]] = {}


def _build_manifest(dataset: str) -> list[dict]:
    key = dataset.upper()
    if key in _manifest_cache:
        return _manifest_cache[key]
    root = _dataset_root(key)
    entries: list[dict] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        name = p.name
        if name.startswith(".") or name == ".DS_Store":
            continue
        entries.append({
            "relpath": p.relative_to(root).as_posix(),
            "size": p.stat().st_size,
        })
    _manifest_cache[key] = entries
    return entries


def _resolve_safe(root: Path, relpath: str) -> Path:
    target = (root / relpath).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=404, detail="path traversal blocked")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return target


_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    m = _RANGE_RE.match(header.strip())
    if not m:
        return None
    start_s, end_s = m.group(1), m.group(2)
    if start_s == "" and end_s == "":
        return None
    if start_s == "":
        n = int(end_s)
        if n == 0:
            return None
        start = max(0, size - n)
        end = size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
    if start > end or start >= size:
        return None
    end = min(end, size - 1)
    return start, end


def _iter_file(path: Path, start: int, end: int) -> Iterator[bytes]:
    remaining = end - start + 1
    with open(path, "rb") as f:
        f.seek(start)
        while remaining > 0:
            chunk = f.read(min(CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


app = FastAPI(title="PRISM raw HSI file server")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/datasets")
def datasets() -> list[str]:
    return [k for k in DATASETS if (REPO_ROOT / DATASETS[k]["raw_root"]).exists()]


@app.get("/manifest/{dataset}")
def manifest(dataset: str) -> list[dict]:
    return _build_manifest(dataset)


@app.get("/download/{dataset}/{relpath:path}")
def download(dataset: str, relpath: str, request: Request):
    root = _dataset_root(dataset)
    target = _resolve_safe(root, relpath)
    size = target.stat().st_size

    range_header = request.headers.get("range")
    if range_header:
        parsed = _parse_range(range_header, size)
        if parsed is None:
            return JSONResponse(
                status_code=416,
                content={"detail": "invalid range"},
                headers={"Content-Range": f"bytes */{size}"},
            )
        start, end = parsed
        length = end - start + 1
        return StreamingResponse(
            _iter_file(target, start, end),
            status_code=206,
            media_type="application/octet-stream",
            headers={
                "Content-Range": f"bytes {start}-{end}/{size}",
                "Content-Length": str(length),
                "Accept-Ranges": "bytes",
            },
        )

    return StreamingResponse(
        _iter_file(target, 0, size - 1),
        status_code=200,
        media_type="application/octet-stream",
        headers={
            "Content-Length": str(size),
            "Accept-Ranges": "bytes",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw HSI file server for ngrok tunnelling.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    available: list[str] = []
    for key in DATASETS:
        root = REPO_ROOT / DATASETS[key]["raw_root"]
        marker = "OK" if root.exists() else "MISSING"
        print(f"  [{marker}] {key:6s} -> {root}")
        if root.exists():
            available.append(key)
    if not available:
        print("ERROR: no dataset raw_root exists on disk; nothing to serve.", file=sys.stderr)
        sys.exit(1)
    print(f"serving datasets: {available}")

    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
