"""
utils/notify_cli.py
-------------------
One-shot Telegram sender for bash-side pings ("grid launched", "grid finished").

Reads the message from --text (or stdin when --text is missing) and emits it
via utils.notify.TelegramNotifier. Silently no-ops when env vars are absent so
missing config never fails the outer bash script.

Usage:
    python utils/notify_cli.py --text "grid launched with 28 runs"
    echo "grid done" | python utils/notify_cli.py
"""

from __future__ import annotations

import argparse
import sys

from utils.notify import TelegramNotifier


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a single Telegram message.")
    parser.add_argument("--text", default=None,
                        help="Message body. If omitted, read from stdin.")
    args = parser.parse_args()

    text = args.text if args.text is not None else sys.stdin.read()
    text = text.strip()
    if not text:
        return 0

    TelegramNotifier().send(text)
    # Always exit 0 — a failed notification must never fail the pipeline.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
