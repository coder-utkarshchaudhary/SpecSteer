"""
scripts/notify_forwarder.py
---------------------------
Runs on the HPC login node inside a tmux window. Tails the notify queue on
shared filesystem and POSTs each JSON line to a local URL — which is the
reverse-tunneled port back to the lab machine's utils/notify_relay.py.

The queue file is populated by training runs on compute nodes: since compute
nodes cannot reach the internet, ``utils.notify.TelegramNotifier`` writes each
outgoing message to ``logs/notify_queue.jsonl`` on the shared home / scratch
filesystem when neither a live relay nor direct Telegram creds are usable.

The forwarder keeps a byte-offset in ``logs/notify_queue.offset`` so a tmux
restart / crash doesn't re-send old messages.

Usage:
    python scripts/notify_forwarder.py \\
        --queue ~/prism/logs/notify_queue.jsonl \\
        --url http://localhost:8765/notify \\
        --interval 3

Idempotency notes:
    * Only lines newline-terminated at read time are consumed; a partial line
      at EOF is retried on the next tick.
    * If the local URL is down, the offset is *not* advanced — we retry the
      same line next tick.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
_LOG = logging.getLogger("notify_forwarder")


def _post(url: str, payload: dict, timeout: float = 10.0) -> bool:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            if 200 <= resp.status < 300:
                return True
            _LOG.warning("POST %s returned %s", url, resp.status)
            return False
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        _LOG.warning("POST %s failed: %s", url, e)
        return False


def _read_offset(offset_path: Path) -> int:
    try:
        return int(offset_path.read_text().strip())
    except (OSError, ValueError):
        return 0


def _write_offset(offset_path: Path, offset: int) -> None:
    try:
        offset_path.write_text(str(offset))
    except OSError as e:
        _LOG.warning("could not persist offset (%s): %s", offset_path, e)


def _drain_once(queue: Path, url: str, offset_path: Path) -> int:
    """One pass over the queue file. Returns number of messages sent."""
    if not queue.is_file():
        return 0

    offset = _read_offset(offset_path)
    size = queue.stat().st_size
    if offset > size:
        # File was rotated / truncated behind our back. Reset.
        _LOG.info("queue shrunk (%d -> %d), resetting offset", offset, size)
        offset = 0

    if offset >= size:
        return 0

    sent = 0
    with queue.open("rb") as fh:
        fh.seek(offset)
        while True:
            line = fh.readline()
            if not line:
                break
            # Partial (unfinished) line — leave for next tick.
            if not line.endswith(b"\n"):
                break
            raw = line.decode("utf-8", errors="replace").strip()
            if not raw:
                offset = fh.tell()
                _write_offset(offset_path, offset)
                continue

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as e:
                _LOG.warning("skipping malformed queue line: %s", e)
                offset = fh.tell()
                _write_offset(offset_path, offset)
                continue

            if not _post(url, payload):
                _LOG.info("relay unavailable; will retry next tick.")
                return sent
            sent += 1
            offset = fh.tell()
            _write_offset(offset_path, offset)

    return sent


def main():
    ap = argparse.ArgumentParser(description="Drain notify_queue.jsonl to a relay URL.")
    ap.add_argument("--queue", required=True, type=Path)
    ap.add_argument("--url", required=True, help="e.g. http://localhost:8765/notify")
    ap.add_argument("--interval", type=float, default=3.0, help="Poll interval (seconds).")
    ap.add_argument("--offset", default=None, type=Path,
                    help="Where to persist the queue read offset. "
                         "Defaults to <queue>.offset next to the queue file.")
    args = ap.parse_args()

    offset_path = args.offset or args.queue.with_suffix(args.queue.suffix + ".offset")
    _LOG.info("forwarder started")
    _LOG.info("  queue    : %s", args.queue)
    _LOG.info("  url      : %s", args.url)
    _LOG.info("  offset   : %s", offset_path)
    _LOG.info("  interval : %.1fs", args.interval)

    try:
        while True:
            n = _drain_once(args.queue, args.url, offset_path)
            if n:
                _LOG.info("forwarded %d message(s)", n)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        _LOG.info("stopping (Ctrl-C).")


if __name__ == "__main__":
    main()
