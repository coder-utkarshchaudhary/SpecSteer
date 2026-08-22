"""
utils/notify.py
---------------
Telegram notifier with three delivery backends, tried in this order per message:

    1. Relay POST — if ``TELEGRAM_RELAY_URL`` is set (HPC case: compute node has
       no internet; the URL is a reverse-tunneled HTTP endpoint on the lab
       machine running ``utils/notify_relay.py``).
    2. Direct Telegram sendMessage — if ``TELEGRAM_BOT_TOKEN`` +
       ``TELEGRAM_CHAT_ID`` are set (lab / cloud case).
    3. On-disk queue — append the payload as one JSON line to
       ``logs/notify_queue.jsonl`` so a forwarder / poller can drain it later.

Every backend is best-effort: failures are logged and swallowed so training is
never gated on the notification path.

The public surface is:

    TelegramNotifier   — low-level send(). Chooses the backend at send-time.
    RunNotifier        — per-run buffer. Records per-epoch metrics, emits a
                         heartbeat every ``log_every`` epochs, and a run-final
                         summary via ``flush_run(status, extra)``.

RunNotifier integration contract in train/train.py:

    notifier = RunNotifier(model, dataset, loss, epochs_planned=N)
    notifier.send_start(config_dict)                            # once, at start
    notifier.record_epoch(epoch, metrics)                       # every epoch
    notifier.mark_best(epoch, monitor)                          # when best ckpt saved
    notifier.heartbeat(epoch, eta_seconds)                      # every log_every epochs
    notifier.flush_run("ok" | "fail" | "early_stop", extra=...) # once, at end

All Telegram messages use ``parse_mode=HTML`` so ``<pre>`` blocks render as
monospace tables. Only ``<``, ``>``, and ``&`` need escaping inside <pre>.
"""

from __future__ import annotations

import html
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False

try:
    import requests
except ImportError:
    requests = None  # type: ignore


_LOG = logging.getLogger(__name__)

_TG_MAX_CHARS = 3800
_QUEUE_PATH = Path("logs") / "notify_queue.jsonl"


def _load_env_once() -> None:
    if not _HAS_DOTENV:
        return
    here = Path(__file__).resolve()
    for candidate in [here.parent.parent / ".env", Path.cwd() / ".env"]:
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return


_load_env_once()


def _write_queue(payload: dict) -> bool:
    """Append one JSON line to logs/notify_queue.jsonl. Best-effort."""
    try:
        _QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _QUEUE_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return True
    except OSError as e:
        _LOG.warning("notify_queue write failed: %s", e)
        return False


class TelegramNotifier:
    """
    Backend-agnostic sender. Picks (in order): relay > direct > queue.
    Failures at each stage fall through to the next.
    """

    API = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, enabled: bool = True):
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        self.relay_url = os.environ.get("TELEGRAM_RELAY_URL", "").strip()
        self._direct_ready = bool(self.token and self.chat_id and requests is not None)
        self._relay_ready = bool(self.relay_url and requests is not None)
        self.enabled = bool(enabled)
        if enabled and not (self._relay_ready or self._direct_ready):
            _LOG.info(
                "Telegram notifier has no live backend (no relay, no direct creds); "
                "messages will queue to %s for a later forwarder.",
                _QUEUE_PATH,
            )

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        chunks = _split_message(text, _TG_MAX_CHARS)
        ok = True
        for chunk in chunks:
            ok &= self._send_one(chunk)
        return ok

    def _send_one(self, text: str) -> bool:
        if self._relay_ready and self._post_relay(text):
            return True
        if self._direct_ready and self._post_direct(text):
            return True
        return _write_queue({
            "chat_id": self.chat_id or None,
            "text": text,
            "parse_mode": "HTML",
            "ts": time.time(),
        })

    def _post_relay(self, text: str) -> bool:
        try:
            resp = requests.post(
                self.relay_url.rstrip("/") + "/notify",
                data=json.dumps({
                    "chat_id": self.chat_id or None,
                    "text": text,
                    "parse_mode": "HTML",
                }),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if resp.status_code // 100 != 2:
                _LOG.warning("relay POST returned %s: %s", resp.status_code, resp.text[:200])
                return False
            return True
        except Exception as e:
            _LOG.warning("relay POST failed: %s", e)
            return False

    def _post_direct(self, text: str) -> bool:
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
        except Exception as e:
            _LOG.warning("Telegram sendMessage failed: %s", e)
            return False


def _split_message(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, buf, length = [], [], 0
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
    train_kld: float
    val_kld: Optional[float]
    # Report-only, validation-side. Defaults keep every existing positional
    # construction valid (is_best was the only defaulted field before).
    val_psnr: Optional[float] = None
    val_ssim: Optional[float] = None
    is_best: bool = False


@dataclass
class RunNotifier:
    model: str
    dataset: str
    loss: str
    log_every: int = 10
    epochs_planned: Optional[int] = None
    seed: Optional[int] = None
    tg: TelegramNotifier = field(default_factory=TelegramNotifier)
    _rows: list[_EpochRow] = field(default_factory=list)
    _latest: Optional[_EpochRow] = None
    _best_epoch: Optional[int] = None
    _best_val: Optional[float] = None
    _start_ts: float = field(default_factory=time.time)
    _last_epoch_seen: int = 0
    # Short, per-process run id. Two concurrent launches of the same cell used
    # to emit byte-identical [HB]/[OK] messages, so a duplicate grid was
    # indistinguishable from the bot misbehaving — which is exactly how the v3
    # double launch went unnoticed until the OOMs started. The id makes two
    # interleaved runs visible at a glance.
    run_id: str = field(default_factory=lambda: f"{os.getpid() % 100000:05d}")

    # ------------------------------------------------------------------

    def _header(self, badge: str) -> str:
        tail = f"  <code>#{self.run_id}</code>"
        if self.seed is not None:
            tail = f" | seed <b>{self.seed}</b>" + tail
        return (
            f"{badge} <b>{html.escape(self.model)}</b> | "
            f"<b>{html.escape(self.dataset)}</b> | <b>{html.escape(self.loss)}</b>"
            + tail
        )

    def send_start(self, config: Optional[dict] = None) -> bool:
        """One-shot 'run started' ping with the resolved hyper-params."""
        lines = [self._header("[START]"), f"host: {html.escape(os.uname().nodename)}"]
        if self.epochs_planned:
            lines.append(f"planned: {self.epochs_planned} epochs")
        if config:
            rows = [f"{k:>18}: {v}" for k, v in sorted(config.items())]
            lines.append("<pre>" + html.escape("\n".join(rows)) + "</pre>")
        return self.tg.send("\n".join(lines))

    # ------------------------------------------------------------------

    def record_epoch(self, epoch: int, metrics: dict) -> None:
        """
        Called every epoch. Stores the row for the trailing heartbeat/summary.
        We keep every row (100 rows * ~120 chars = 12 KB) so the finish message
        can render a stride-10 table.
        """
        self._last_epoch_seen = epoch
        row = _EpochRow(
            epoch=epoch,
            train_loss=float(metrics.get("train_loss", float("nan"))),
            val_loss=_maybe_float(metrics.get("val_loss")),
            train_mse=float(metrics.get("train_mse", float("nan"))),
            val_mse=_maybe_float(metrics.get("val_mse")),
            train_sam=float(metrics.get("train_sam", float("nan"))),
            val_sam=_maybe_float(metrics.get("val_sam")),
            train_kld=float(metrics.get("train_kld", float("nan"))),
            val_kld=_maybe_float(metrics.get("val_kld")),
            val_psnr=_maybe_float(metrics.get("val_psnr")),
            val_ssim=_maybe_float(metrics.get("val_ssim")),
        )
        self._rows.append(row)
        self._latest = row

    def mark_best(self, epoch: int, monitor_value: float) -> None:
        self._best_epoch = epoch
        self._best_val = float(monitor_value)

    # ------------------------------------------------------------------

    def heartbeat(self, epoch: int, eta_seconds: Optional[float] = None) -> bool:
        """
        Fire the periodic per-N-epoch check-in. Called *after* record_epoch so
        ``self._latest`` reflects the current epoch's metrics.
        """
        if epoch == 0 or epoch % self.log_every != 0:
            return False
        if self._latest is None:
            return False

        elapsed = time.time() - self._start_ts
        eta_str = _fmt_hms(eta_seconds) if eta_seconds is not None else "n/a"
        planned = self.epochs_planned or "?"

        lines = [
            self._header("[HB]"),
            f"epoch {epoch}/{planned}  ·  wall {_fmt_hms(elapsed)}  ·  ETA {eta_str}",
        ]

        r = self._latest
        rows = [
            f"{'':<8}{'train':>12}{'val':>12}",
            f"{'loss':<8}{r.train_loss:>12.6f}{_fmt(r.val_loss, 12, 6):>12}",
            f"{'mse':<8}{r.train_mse:>12.6f}{_fmt(r.val_mse, 12, 6):>12}",
            f"{'sam':<8}{r.train_sam:>12.6f}{_fmt(r.val_sam, 12, 6):>12}",
            f"{'kld':<8}{r.train_kld:>12.6f}{_fmt(r.val_kld, 12, 6):>12}",
            # Report-only, val-side; comparable across all four models in a way
            # `mse` is not (vae-our's `mse` is a 3-branch average).
            f"{'psnr':<8}{'-':>12}{_fmt(r.val_psnr, 12, 2):>12}",
            f"{'ssim':<8}{'-':>12}{_fmt(r.val_ssim, 12, 4):>12}",
        ]
        lines.append("<pre>" + html.escape("\n".join(rows)) + "</pre>")

        if self._best_epoch is not None and self._best_val is not None:
            lines.append(
                f"best SAM so far: epoch {self._best_epoch} "
                f"(val_sam={self._best_val:.6f})"
            )
        return self.tg.send("\n".join(lines))

    # ------------------------------------------------------------------

    def flush_run(self, status: str = "ok", extra: str = "") -> bool:
        """Emit the summary message. status: 'ok', 'fail', or 'early_stop'."""
        badge = {"ok": "[OK]", "fail": "[FAIL]", "early_stop": "[STOP]"}.get(
            status, f"[{status.upper()}]"
        )

        if self._best_epoch is not None:
            for row in self._rows:
                if row.epoch == self._best_epoch:
                    row.is_best = True
                    break

        elapsed = time.time() - self._start_ts
        planned = self.epochs_planned or "?"

        header_lines = [
            self._header(badge),
            f"epochs {self._last_epoch_seen}/{planned}  ·  wall {_fmt_hms(elapsed)}",
        ]

        try:
            import torch
            if torch.cuda.is_available():
                peak_bytes = torch.cuda.max_memory_allocated()
                header_lines[-1] += f"  ·  peak GPU {peak_bytes / (1024**3):.1f} GB"
        except Exception:
            pass

        if self._best_epoch is not None and self._best_val is not None:
            header_lines.append(
                f"best @ epoch {self._best_epoch}: {self._best_val:.6f} (checkpoint saved)"
            )

        # Stride-log table: every log_every epoch + the very last epoch.
        stride_rows = [
            r for r in self._rows
            if (r.epoch % self.log_every == 0) or r.epoch == self._last_epoch_seen
        ]
        table = _format_table(stride_rows) if stride_rows else "(no epochs completed)"

        body = "\n".join(header_lines) + f"\n<pre>{html.escape(table)}</pre>"
        if extra:
            body += f"\n<pre>{html.escape(extra[-1500:])}</pre>"

        return self.tg.send(body)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _maybe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt_hms(seconds: Optional[float]) -> str:
    if seconds is None or seconds != seconds:  # NaN guard
        return "n/a"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def _fmt(v: Optional[float], width: int = 9, prec: int = 4) -> str:
    if v is None:
        return "-".rjust(width)
    return f"{v:>{width}.{prec}f}"


def _format_table(rows: list[_EpochRow]) -> str:
    """
    Stride-10 summary table.

    The train-side mse/sam columns were dropped when PSNR and SSIM were added:
    at seven columns the header was already ~75 characters and Telegram wraps a
    <pre> block past roughly that on mobile, which makes the table unreadable
    rather than merely wide. The val columns are the ones a summary is read for;
    tr_loss stays as the overfitting tell.
    """
    has_val = any(r.val_loss is not None for r in rows)
    if has_val:
        header = (
            f"{'ep':>4}  {'tr_loss':>9}  {'val_loss':>9}  {'val_mse':>9}  "
            f"{'val_sam':>8}  {'psnr':>7}  {'ssim':>7}"
        )
    else:
        header = f"{'ep':>4}  {'tr_loss':>9}  {'tr_mse':>9}  {'tr_sam':>8}"

    lines = [header, "-" * len(header)]
    for r in rows:
        marker = " *" if r.is_best else "  "
        if has_val:
            line = (
                f"{r.epoch:>4}  {r.train_loss:>9.4f}  {_fmt(r.val_loss):>9}  "
                f"{_fmt(r.val_mse):>9}  {_fmt(r.val_sam):>8}  "
                f"{_fmt(r.val_psnr, 7, 2):>7}  {_fmt(r.val_ssim, 7, 4):>7}"
            )
        else:
            line = f"{r.epoch:>4}  {r.train_loss:>9.4f}  {r.train_mse:>9.4f}  {r.train_sam:>8.4f}"
        lines.append(line + marker)
    if any(r.is_best for r in rows):
        lines.append("(*) new best val SAM checkpoint")
    return "\n".join(lines)
