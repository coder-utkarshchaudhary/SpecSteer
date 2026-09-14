"""
utils/notify_cli.py
-------------------
One-shot Telegram sender for bash-side pings ("grid launched", "grid finished")
and, with --file, for sending a local file (e.g. a results CSV) as a document.

Reads the message from --text (or stdin when --text is missing) and emits it
via utils.notify.TelegramNotifier. Silently no-ops when env vars are absent so
missing config never fails the outer bash script.

Usage:
    python utils/notify_cli.py --text "grid launched with 28 runs"
    echo "grid done" | python utils/notify_cli.py
    python utils/notify_cli.py --file results/ablation_table.csv --caption "IIRS done"
"""

from __future__ import annotations

import argparse
import sys

from utils.notify import TelegramNotifier


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a single Telegram message or file.")
    parser.add_argument("--text", default=None,
                        help="Message body. If omitted (and --file is not given), "
                             "read from stdin.")
    parser.add_argument("--file", default=None,
                        help="Path to a local file to send as a document, "
                             "instead of a text message. A missing or empty "
                             "file is silently skipped.")
    parser.add_argument("--caption", default="",
                        help="Caption for --file (ignored otherwise).")
    args = parser.parse_args()

    if args.file is not None:
        from pathlib import Path
        path = Path(args.file)
        if path.is_file() and path.stat().st_size > 0:
            TelegramNotifier().send_document(path, caption=args.caption)
        # Always exit 0 — a failed/missing/empty file must never fail the pipeline.
        return 0

    text = args.text if args.text is not None else sys.stdin.read()
    text = text.strip()
    if not text:
        return 0

    TelegramNotifier().send(text)
    # Always exit 0 — a failed notification must never fail the pipeline.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
