"""
utils/logging_setup.py
----------------------
Shared file-based logging for the HSI VAE ablation entrypoints.

Every training / inference / downstream invocation writes a self-contained log
under ``logs/`` at the repo root. Filenames mirror the checkpoint naming rule
in ``modules.registry.checkpoint_name`` so a log lines up 1:1 with the
checkpoint it produced:

    train_vae-our_<dataset>_<ts>.log                (physics-only model)
    train_<model>_<loss>_<dataset>_<ts>.log         (baselines)
    inference_...                                   (same rules)
    downstream_<model>_<dataset>_<ts>.log           (one per model)

Level policy for training:
    - DEBUG for the first ``debug_epochs`` epochs (default 3), then INFO.
    - ``set_epoch(n)`` on the returned logger flips the level on both the
      file and stream handlers atomically.
    - ``log_tensor(logger, name, tensor)`` prints shape/dtype/device only —
      never element values — so DEBUG logs stay small and safe.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from modules.registry import PHYSICS_ONLY


LOG_DIR = Path("logs")


def timestamp() -> str:
    """Return a filesystem-safe timestamp string, e.g. '20260722-143012'."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_log_name(
    kind: str,
    model: str,
    dataset: str,
    loss: Optional[str] = None,
    ts: Optional[str] = None,
) -> str:
    if ts is None:
        ts = timestamp()
    include_loss = loss is not None and model not in PHYSICS_ONLY
    parts = [kind, model]
    if include_loss:
        parts.append(loss)
    parts.append(dataset)
    parts.append(ts)
    return "_".join(parts) + ".log"


_FORMATTER = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


class _EpochAwareLogger(logging.Logger):
    """Logger with a helper to flip DEBUG↔INFO at an epoch boundary."""

    debug_epochs: int = 3

    def set_epoch(self, epoch: int) -> None:
        """Called by the training loop at the start of each epoch."""
        want = logging.DEBUG if epoch < self.debug_epochs else logging.INFO
        if self.level != want:
            self.setLevel(want)
            for h in self.handlers:
                h.setLevel(want)
            self.info(
                "log level -> %s (epoch %d, debug_epochs=%d)",
                logging.getLevelName(want), epoch, self.debug_epochs,
            )


def get_run_logger(
    kind: str,
    model: str,
    dataset: str,
    loss: Optional[str] = None,
    ts: Optional[str] = None,
    stream: bool = True,
    debug_epochs: int = 3,
) -> _EpochAwareLogger:
    """
    Return an ``_EpochAwareLogger`` for one run. Starts at DEBUG so epoch 0
    setup already emits shapes; the trainer calls ``.set_epoch(e)`` at each
    epoch to switch to INFO once ``e >= debug_epochs``.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    filename = run_log_name(kind, model, dataset, loss=loss, ts=ts)
    log_path = LOG_DIR / filename

    # Register our subclass so getLogger() returns _EpochAwareLogger instances.
    logging.setLoggerClass(_EpochAwareLogger)
    logger = logging.getLogger(log_path.stem)
    # Reset class back so we don't affect unrelated loggers.
    logging.setLoggerClass(logging.Logger)

    logger.debug_epochs = debug_epochs
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(_FORMATTER)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        if stream:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(_FORMATTER)
            stream_handler.setLevel(logging.DEBUG)
            logger.addHandler(stream_handler)

    logger.info("log file: %s", log_path)
    return logger  # type: ignore[return-value]


def get_console_logger(name: str = "specsteer.console") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(_FORMATTER)
        logger.addHandler(stream_handler)
    return logger


def log_tensor(logger: logging.Logger, name: str, t) -> None:
    """
    Emit a DEBUG-only line describing a tensor by its metadata alone.

    Never prints element values — safe to leave in hot code paths without
    risking log bloat or leaking numerical content.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    try:
        shape = tuple(t.shape)
        dtype = getattr(t, "dtype", "?")
        device = getattr(t, "device", "?")
    except Exception:
        logger.debug("tensor %s: (unreadable metadata)", name)
        return
    logger.debug("tensor %s: shape=%s dtype=%s device=%s", name, shape, dtype, device)


def log_tensors(logger: logging.Logger, **named) -> None:
    """Convenience wrapper: ``log_tensors(log, x=x, z=z)``."""
    for k, v in named.items():
        log_tensor(logger, k, v)


def get_log_path(logger: logging.Logger) -> Optional[Path]:
    """Return the file path of the first FileHandler attached, if any."""
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            return Path(h.baseFilename)
    return None


def tail_log(logger: logging.Logger, n_lines: int = 40) -> str:
    """Read the last ``n_lines`` from the logger's file handler. Best-effort."""
    p = get_log_path(logger)
    if p is None or not p.is_file():
        return ""
    try:
        with p.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            block = min(size, 32 * 1024)
            fh.seek(size - block)
            data = fh.read().decode("utf-8", errors="replace")
        lines = data.splitlines()[-n_lines:]
        return "\n".join(lines)
    except OSError:
        return ""
