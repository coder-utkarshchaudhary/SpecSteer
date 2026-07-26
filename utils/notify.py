"""
utils/notify.py
---------------
Best-effort Telegram notifier for the overnight ablation grid.

Two classes:

    TelegramNotifier   — thin HTTP wrapper around sendMessage. Reads
                         TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from the
                         environment (or a .env file at repo root if
                         python-dotenv is installed). Never raises on network
                         or auth failure — those degrade to a single logger
                         warning so training is never gated on Telegram.

    RunNotifier        — per-run buffer. Records per-epoch metrics, keeps only
                         every ``log_every``-th epoch, tracks the best epoch,
                         and emits one summary message per training run via
                         ``flush_run(...)``.

Contract for training integration:

    notifier = RunNotifier(model, dataset, loss)          # in main()
    notifier.record_epoch(epoch, metrics)                 # after each epoch log
    notifier.mark_best(epoch, monitor)                    # when a new best ckpt saved
    notifier.flush_run("ok" | "fail" | "early_stop",      # once on run end
                       extra="optional footer text")

The message is a self-contained HTML block (parse_mode="HTML") so tables render
as monospace and the string escaping surface is minimal (only <, >, &).
"""

from __future__ import annotations

import html
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:  # optional; if missing we fall back to plain os.environ.
    from dotenv import load_dotenv
    _HAS_DOTENV = True
except ImportError:  # pragma: no cover - handled by requirements.txt at deploy
    _HAS_DOTENV = False

try:
    import requests
except ImportError:  # pragma: no cover - requests is already required
    requests = None  # type: ignore


_LOG = logging.getLogger(__name__)

# Telegram caps a single sendMessage at 4096 characters; leave headroom for the
# <pre>...</pre> wrapper we emit around tables.
_TG_MAX_CHARS = 3800


def _load_env_once() -> None:
    """Load repo-root .env once per process, silently if dotenv is absent."""
    if not _HAS_DOTENV:
        return
    # Walk up from this file to find a .env at the repo root; fall back to CWD.
    here = Path(__file__).resolve()
    for candidate in [here.parent.parent / ".env", Path.cwd() / ".env"]:
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return


_load_env_once()


class TelegramNotifier:
    """
    Minimal Telegram Bot API client. Best-effort: any failure is logged and
    swallowed. Callers can pass ``enabled=False`` (or omit env vars) to
    silently no-op.
    """

    API = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, enabled: bool = True):
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        self.enabled = bool(enabled and self.token and self.chat_id and requests is not None)
        if enabled and not self.enabled:
            missing = []
            if not self.token:
                missing.append("TELEGRAM_BOT_TOKEN")
            if not self.chat_id:
                missing.append("TELEGRAM_CHAT_ID")
            if requests is None:
                missing.append("requests package")
            _LOG.warning(
                "Telegram notifier disabled — missing: %s. "
                "Training will continue; on-disk logs remain the source of truth.",
                ", ".join(missing) or "(unknown)",
            )

    def send(self, text: str) -> bool:
        """
        POST ``text`` to Telegram. Returns True on success, False on any
        failure. Splits messages > _TG_MAX_CHARS into consecutive parts so
        long summaries don't get rejected.
        """
        if not self.enabled:
            return False
        for chunk in _split_message(text, _TG_MAX_CHARS):
            ok = self._send_one(chunk)
            if not ok:
                return False
        return True

    def _send_one(self, text: str) -> bool:
        try:
            resp = requests.post(
                self.API.format(token=self.token),
                data={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": "true",
                },
                timeout=10,
            )
            if resp.status_code != 200:
                _LOG.warning(
                    "Telegram sendMessage returned %s: %s",
                    resp.status_code, resp.text[:200],
                )
                return False
            return True
        except Exception as e:  # noqa: BLE001 — best-effort by design
            _LOG.warning("Telegram sendMessage failed: %s", e)
            return False


def _split_message(text: str, limit: int) -> list[str]:
    """Split a message on line boundaries so each chunk fits under ``limit``."""
    if len(text) <= limit:
        return [text]
    parts, buf = [], []
    length = 0
    for line in text.splitlines(keepends=True):
        if length + len(line) > limit and buf:
            parts.append("".join(buf))
            buf, length = [], 0
        buf.append(line)
        length += len(line)
    if buf:
        parts.append("".join(buf))
    return parts


# ---------------------------------------------------------------------------
# Per-run buffer
# ---------------------------------------------------------------------------

@dataclass
class _EpochRow:
    epoch: int
    train_loss: float
    val_loss: Optional[float]
    train_mse: float
    val_mse: Optional[float]
    train_sam: float
    val_sam: Optional[float]
    is_best: bool = False


@dataclass
class RunNotifier:
    """
    One RunNotifier per (model, dataset, loss) run. Buffers per-10-epoch rows
    and sends a single Telegram message at run end.

    Metrics passed to ``record_epoch`` are a flat dict; ``flush_run`` renders
    them as a monospace table.
    """

    model: str
    dataset: str
    loss: str
    log_every: int = 10
    epochs_planned: Optional[int] = None
    tg: TelegramNotifier = field(default_factory=TelegramNotifier)
    _rows: list[_EpochRow] = field(default_factory=list)
    _best_epoch: Optional[int] = None
    _best_val: Optional[float] = None
    _start_ts: float = field(default_factory=time.time)
    _last_epoch_seen: int = 0

    def record_epoch(self, epoch: int, metrics: dict) -> None:
        """
        Append the current epoch to the buffer if it lies on the ``log_every``
        grid. ``metrics`` should contain: train_loss, train_mse, train_sam, and
        the ``val_*`` variants when a val loader was used.
        """
        self._last_epoch_seen = epoch
        if epoch % self.log_every != 0:
            return
        self._rows.append(_EpochRow(
            epoch=epoch,
            train_loss=float(metrics.get("train_loss", float("nan"))),
            val_loss=_maybe_float(metrics.get("val_loss")),
            train_mse=float(metrics.get("train_mse", float("nan"))),
            val_mse=_maybe_float(metrics.get("val_mse")),
            train_sam=float(metrics.get("train_sam", float("nan"))),
            val_sam=_maybe_float(metrics.get("val_sam")),
        ))

    def mark_best(self, epoch: int, monitor_value: float) -> None:
        """Called every time train.py saves a new best checkpoint."""
        self._best_epoch = epoch
        self._best_val = float(monitor_value)

    def flush_run(self, status: str = "ok", extra: str = "") -> bool:
        """
        Emit the summary Telegram message. ``status`` is 'ok', 'fail', or
        'early_stop'; ``extra`` is appended as a footer (e.g. traceback tail
        on failure).
        """
        # Retro-mark the best row (if one exists inside the buffered set).
        if self._best_epoch is not None:
            for row in self._rows:
                if row.epoch == self._best_epoch:
                    row.is_best = True
                    break

        elapsed_s = time.time() - self._start_ts
        status_emoji = {"ok": "OK", "fail": "FAIL", "early_stop": "STOP"}.get(status, status.upper())

        header_lines = [
            f"<b>{status_emoji}</b> {html.escape(self.model)} | "
            f"{html.escape(self.dataset)} | {html.escape(self.loss)}",
            f"epochs seen: {self._last_epoch_seen}"
            + (f"/{self.epochs_planned}" if self.epochs_planned else "")
            + f"  |  wall: {_fmt_hms(elapsed_s)}",
        ]

        if self._best_epoch is not None and self._best_val is not None:
            header_lines.append(
                f"best @ epoch {self._best_epoch}: {self._best_val:.6f} (checkpoint saved)"
            )

        table = _format_table(self._rows) if self._rows else "(no epochs completed)"
        body = "\n".join(header_lines) + f"\n<pre>{html.escape(table)}</pre>"
        if extra:
            body += f"\n<pre>{html.escape(extra[-1500:])}</pre>"

        return self.tg.send(body)


def _maybe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt_hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def _format_table(rows: list[_EpochRow]) -> str:
    """
    Render buffered rows as a fixed-width table. Columns adapt to whether the
    run had a validation loader (val_* columns are hidden if all val entries
    are None).
    """
    has_val = any(r.val_loss is not None for r in rows)
    if has_val:
        header = f"{'ep':>4}  {'tr_loss':>9}  {'val_loss':>9}  {'tr_mse':>9}  {'val_mse':>9}  {'tr_sam':>8}  {'val_sam':>8}"
    else:
        header = f"{'ep':>4}  {'tr_loss':>9}  {'tr_mse':>9}  {'tr_sam':>8}"

    lines = [header, "-" * len(header)]
    for r in rows:
        marker = " *" if r.is_best else "  "
        if has_val:
            line = (
                f"{r.epoch:>4}  {r.train_loss:>9.4f}  "
                f"{_fmt(r.val_loss):>9}  {r.train_mse:>9.4f}  "
                f"{_fmt(r.val_mse):>9}  {r.train_sam:>8.4f}  "
                f"{_fmt(r.val_sam):>8}"
            )
        else:
            line = f"{r.epoch:>4}  {r.train_loss:>9.4f}  {r.train_mse:>9.4f}  {r.train_sam:>8.4f}"
        lines.append(line + marker)
    if any(r.is_best for r in rows):
        lines.append("(*) new best val checkpoint")
    return "\n".join(lines)


def _fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.4f}"
