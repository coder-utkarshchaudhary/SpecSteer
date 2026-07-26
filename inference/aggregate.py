"""
inference/aggregate.py
----------------------
Roll up the per-cell reconstruction metrics from ``results/inference/*.json``
and the per-dataset downstream results from
``results/downstream/<DATASET>/downstream_results.json`` into two CSVs and
(optionally) a single Telegram summary.

Run from the repo root with PYTHONPATH set:
    PYTHONPATH=. python inference/aggregate.py --telegram

Inputs:
    --inference-dir   directory of per-run inference JSONs (default: results/inference)
    --downstream-dir  directory containing <DATASET>/downstream_results.json
                      subfolders (default: results/downstream)

Outputs:
    <out-dir>/ablation_table.csv    — one row per (dataset, model, loss)
    <out-dir>/downstream_table.csv  — one row per (dataset, model), summarising
                                      noise-injection robustness + interp
    stdout                          — both tables in fixed-width form
    Telegram                        — same summary (best-effort, --telegram flag)
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Optional

from utils.notify import TelegramNotifier


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_inference_rows(inference_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not inference_dir.is_dir():
        return rows
    for path in sorted(inference_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"[warn] failed to parse {path}: {e}")
            continue
        rows.append(data)
    rows.sort(key=lambda r: (r.get("dataset", ""), r.get("model", ""), r.get("loss", "")))
    return rows


def load_downstream_rows(downstream_dir: Path) -> list[dict]:
    """
    One row per (dataset, model). Summary fields extracted from the noise sweep:
      - psnr_clean : PSNR at sigma=0
      - psnr_mid   : PSNR at sigma=0.5
      - psnr_drop  : psnr_clean - psnr_mid  (larger = more fragile)
      - sam_mid    : SAM at sigma=0.5 (rad)
    And from the interpolation experiment:
      - jaggedness, path_length
    """
    out: list[dict] = []
    if not downstream_dir.is_dir():
        return out
    for dataset_dir in sorted(downstream_dir.iterdir()):
        summary_file = dataset_dir / "downstream_results.json"
        if not summary_file.is_file():
            continue
        try:
            entries = json.loads(summary_file.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"[warn] failed to parse {summary_file}: {e}")
            continue

        for entry in entries:
            noise = {round(d["sigma"], 4): d for d in entry.get("noise", [])}
            interp = entry.get("interp", {}) or {}
            row = {
                "dataset": dataset_dir.name,
                "model": entry.get("model", "?"),
                "loss": entry.get("loss", "?"),
                "psnr_clean": _get(noise.get(0.0), "psnr"),
                "psnr_mid": _get(noise.get(0.5), "psnr"),
                "psnr_drop": _diff(noise.get(0.0), noise.get(0.5), "psnr"),
                "sam_mid": _get(noise.get(0.5), "sam"),
                "jaggedness": interp.get("jaggedness"),
                "path_length": interp.get("path_length"),
            }
            out.append(row)
    out.sort(key=lambda r: (r["dataset"], r["model"]))
    return out


def _get(d: Optional[dict], k: str) -> Optional[float]:
    return None if d is None else d.get(k)


def _diff(a: Optional[dict], b: Optional[dict], k: str) -> Optional[float]:
    va, vb = _get(a, k), _get(b, k)
    return None if va is None or vb is None else va - vb


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

INFERENCE_COLS = ["dataset", "model", "loss", "mse", "sam_rad", "psnr", "ssim", "n_samples"]
DOWNSTREAM_COLS = ["dataset", "model", "loss", "psnr_clean", "psnr_mid",
                   "psnr_drop", "sam_mid", "jaggedness", "path_length"]


def write_csv(rows: list[dict], cols: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c) for c in cols})


def render_table(rows: list[dict], cols: list[str], title: str) -> str:
    if not rows:
        return f"{title}\n(no rows)\n"

    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    widths = {c: max(len(c), max((len(fmt(r.get(c))) for r in rows), default=0)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    line = "-" * len(header)
    body_lines = ["  ".join(fmt(r.get(c)).ljust(widths[c]) for c in cols) for r in rows]
    return f"{title}\n{header}\n{line}\n" + "\n".join(body_lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate inference + downstream results.")
    p.add_argument("--inference-dir", default="results/inference", type=Path)
    p.add_argument("--downstream-dir", default="results/downstream", type=Path)
    p.add_argument("--out-dir", default="results", type=Path,
                   help="Where the two CSVs are written.")
    p.add_argument("--telegram", action="store_true",
                   help="Also send a summary message to Telegram.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    inference_rows = load_inference_rows(args.inference_dir)
    downstream_rows = load_downstream_rows(args.downstream_dir)

    inference_csv = args.out_dir / "ablation_table.csv"
    downstream_csv = args.out_dir / "downstream_table.csv"
    write_csv(inference_rows, INFERENCE_COLS, inference_csv)
    write_csv(downstream_rows, DOWNSTREAM_COLS, downstream_csv)

    inference_table = render_table(
        inference_rows, INFERENCE_COLS,
        f"Reconstruction metrics ({len(inference_rows)} cells)",
    )
    downstream_table = render_table(
        downstream_rows, DOWNSTREAM_COLS,
        f"Downstream latent probes ({len(downstream_rows)} cells)",
    )

    print(inference_table)
    print(downstream_table)
    print(f"CSV: {inference_csv}")
    print(f"CSV: {downstream_csv}")

    if args.telegram:
        summary = (
            f"<b>Inference sweep finished</b>\n"
            f"reconstruction cells: {len(inference_rows)}  |  "
            f"downstream cells: {len(downstream_rows)}\n"
            f"<pre>{html.escape(inference_table)}</pre>\n"
            f"<pre>{html.escape(downstream_table)}</pre>\n"
            f"CSVs: {inference_csv} , {downstream_csv}"
        )
        TelegramNotifier().send(summary)


if __name__ == "__main__":
    main()
