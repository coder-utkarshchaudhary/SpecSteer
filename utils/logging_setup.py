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

``ts`` is captured once by the caller (``timestamp()``) and threaded through so
the filename suffix matches the banner inside the log.
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
    """
    Build the log filename for one run.

    Physics-only models drop the loss token (they have no meaningful alternative
    regime), matching ``modules.registry.checkpoint_name``.
    """
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


def get_run_logger(
    kind: str,
    model: str,
    dataset: str,
    loss: Optional[str] = None,
    ts: Optional[str] = None,
    stream: bool = True,
) -> logging.Logger:
    """
    Return a logger that writes to ``logs/<run_log_name(...)>``.

    With ``stream=True`` (default), progress is also mirrored to stdout — this
    is what train/inference want. Downstream runs many models per invocation
    and pairs several file-only loggers (``stream=False``) with a single shared
    ``get_console_logger()`` so shared messages are broadcast to every file but
    only printed once.

    Handlers are attached once per logger name; calling this repeatedly with
    the same args reuses the same logger without duplicating output.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    filename = run_log_name(kind, model, dataset, loss=loss, ts=ts)
    log_path = LOG_DIR / filename

    logger = logging.getLogger(log_path.stem)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(_FORMATTER)
        logger.addHandler(file_handler)
        if stream:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(_FORMATTER)
            logger.addHandler(stream_handler)

    return logger


def get_console_logger(name: str = "specsteer.console") -> logging.Logger:
    """
    Return a stdout-only logger. Used by downstream to print shared messages
    (banner, comparison tables) once while its per-model file loggers persist
    the same messages to every log file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(_FORMATTER)
        logger.addHandler(stream_handler)
    return logger
