"""
utils/notify_relay.py
---------------------
Lab-side HTTP relay for the HPC pipeline.

The IITD HPC compute nodes have no outbound internet, so training cannot reach
Telegram directly. Instead, ``scripts/hpc_launch.sh`` opens a reverse SSH
tunnel from the HPC login node back to this relay, and a forwarder on the
login node (``scripts/notify_forwarder.py``) drains a shared-FS queue file
by POSTing each line to ``http://localhost:<port>/notify`` — which is really
this process on the lab machine.

Endpoints (single small ``http.server``):

    POST /notify     JSON body: {chat_id?, text, parse_mode?}
                     -> forwards to Telegram via TelegramNotifier

    GET  /healthz    -> "ok"
    GET  /           -> short usage note

Run from the repo root:

    python utils/notify_relay.py --port 8765
    python utils/notify_relay.py --port 8765 --bind 127.0.0.1

Env: reads TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from .env / environment.
Bind defaults to 127.0.0.1 so only the reverse tunnel can reach it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.notify import TelegramNotifier  # noqa: E402


LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "relay.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
_LOG = logging.getLogger("notify_relay")


# One shared TelegramNotifier — reads env once at import.
_TG = TelegramNotifier(enabled=True)


class _RelayHandler(BaseHTTPRequestHandler):
    server_version = "SpecsteerRelay/1.0"

    def log_message(self, fmt, *args):
        _LOG.info("%s - " + fmt, self.address_string(), *args)

    def _respond(self, status: int, body: str, ctype: str = "text/plain") -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        if self.path == "/healthz":
            return self._respond(200, "ok")
        if self.path == "/":
            return self._respond(
                200,
                "specsteer notify_relay\nPOST /notify {chat_id?, text, parse_mode?}\n",
            )
        self._respond(404, "not found")

    def do_POST(self):  # noqa: N802
        if self.path != "/notify":
            return self._respond(404, "not found")

        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return self._respond(400, "empty body")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            _LOG.warning("bad JSON: %s", e)
            return self._respond(400, "bad json")

        text = payload.get("text")
        if not text:
            return self._respond(400, "missing 'text'")

        # If the client wants a specific chat_id we honour it; else fall back
        # to whatever TelegramNotifier picked up from env.
        chat_id_override = payload.get("chat_id")
        if chat_id_override:
            _TG.chat_id = str(chat_id_override).strip() or _TG.chat_id

        _LOG.info("relaying message (%d chars)", len(text))
        ok = _TG.send(text)
        if ok:
            self._respond(200, "sent")
        else:
            self._respond(502, "telegram send failed")


def main():
    ap = argparse.ArgumentParser(description="Lab-side Telegram relay for HPC.")
    ap.add_argument("--port", type=int, default=int(os.environ.get("LAB_TUNNEL_PORT", "8765")))
    ap.add_argument("--bind", default="127.0.0.1",
                    help="Bind address (default 127.0.0.1 — only the reverse tunnel can reach it).")
    args = ap.parse_args()

    if not _TG.enabled:
        _LOG.warning(
            "TelegramNotifier is disabled at import (missing TELEGRAM_BOT_TOKEN / "
            "TELEGRAM_CHAT_ID?). Relay will accept POSTs but Telegram sends will "
            "fall through to the on-disk queue."
        )

    srv = ThreadingHTTPServer((args.bind, args.port), _RelayHandler)
    _LOG.info("notify_relay listening on http://%s:%d/", args.bind, args.port)
    _LOG.info("healthcheck: curl http://%s:%d/healthz", args.bind, args.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        _LOG.info("shutting down (Ctrl-C).")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
