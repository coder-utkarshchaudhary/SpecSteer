"""
visualisations/iclr/make_radar_charts.py
-----------------------------------------
Seed-averaged radar ("spider") charts, one per dataset, all 5 grid models,
axes SAM / PSNR / SSIM, normalized 0-1 within each dataset for the radar
geometry (tick labels show the original values).

Styled to match the four PNGs already shared ad hoc (~/Downloads/*_radar.png)
-- no generator script for those exists anywhere in this repo (confirmed by
search), so this is a from-scratch reimplementation of that same style, with
seed-averaging added on top. Legend labels and dataset-key titles (CRIMS, not
CRISM) are a deliberate match to those PNGs, not the paper's prose spelling.

Usage
=====
    PYTHONPATH=. python visualisations/iclr/make_radar_charts.py
    PYTHONPATH=. python visualisations/iclr/make_radar_charts.py \\
        --input "ChatExport_2026-09-18/ablation_table (6).csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# Registry key -> legend label, matching the existing Downloads PNGs exactly.
# No display-name mapping exists anywhere in the codebase (confirmed by
# search) -- this is the only place these strings live.
MODEL_LABELS = {
    "vae-1d-pixelwise": "VAE 1D Pixelwise",
    "vae-3d-spatio-spectral": "VAE 3D Spatio-Spectral",
    "vae-our": "VAE Ours",
    "vae-our-nl": "VAE Ours + NL",
    "vae-standard": "VAE Standard",
}
# Draw order / color, matching the existing Downloads PNGs (matplotlib's
# default tab10 cycle in this same order: blue, orange, green, red, purple).
MODEL_ORDER = ["vae-1d-pixelwise", "vae-3d-spatio-spectral", "vae-our",
               "vae-our-nl", "vae-standard"]

DATASET_ORDER = ["IIRS", "AVIRIS", "M3", "CRIMS"]

METRICS = ["sam_valid", "psnr", "ssim"]
METRIC_LABELS = {"sam_valid": "SAM", "psnr": "PSNR", "ssim": "SSIM"}
# SAM is "lower is better" -- inverted during normalization so every axis
# reads "further out = better", matching the existing charts' convention.
INVERT = {"sam_valid": True, "psnr": False, "ssim": False}


def load_seed_averaged(input_path: Path) -> pd.DataFrame:
    """
    Filter to the SAM-selected checkpoint and the physics loss regime, then
    average sam_valid/psnr/ssim across seeds.

    Two filters matter and are silent failure modes if skipped:
      - select == "sam": ablation_table.csv carries a few extra
        select == "mse" rows (currently M3's vae-our/vae-our-nl only). Without
        this filter those two models' M3 seed-average would silently pool
        over 4 rows instead of 2 while every other cell pools over 2.
      - loss == "physics": vae-our/vae-our-nl only have physics rows; the
        three baselines have both regimes. Physics-loss rows are used for
        every model so the comparison stays like-for-like (the grid's own
        framing of the physics-loss comparison as "the claim" being tested).
    """
    df = pd.read_csv(input_path)
    df = df[(df["select"] == "sam") & (df["loss"] == "physics")]
    missing = set(MODEL_ORDER) - set(df["model"].unique())
    if missing:
        raise SystemExit(f"No physics/select=sam rows for: {sorted(missing)} "
                          f"in {input_path} -- check the input file.")
    grouped = (df.groupby(["dataset", "model"])[METRICS]
                 .mean()
                 .reset_index())
    counts = df.groupby(["dataset", "model"]).size()
    off = counts[counts != 2]
    if len(off):
        print(f"NOTE: non-2-seed groups (averaged anyway, but check these):\n{off}")
    return grouped


def normalize_for_radar(grouped: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Per-dataset min-max normalize each metric to [0, 1]; invert SAM."""
    sub = grouped[grouped["dataset"] == dataset].set_index("model").reindex(MODEL_ORDER)
    norm = pd.DataFrame(index=sub.index)
    for m in METRICS:
        vals = sub[m].astype(float)
        lo, hi = vals.min(), vals.max()
        span = hi - lo if hi > lo else 1.0
        n = (vals - lo) / span
        norm[m] = (1.0 - n) if INVERT[m] else n
        norm[f"{m}_raw"] = vals
    return norm


def plot_dataset(norm: pd.DataFrame, dataset: str, out_path: Path) -> None:
    n_axes = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    palette = sns.color_palette("tab10", n_colors=len(MODEL_ORDER))

    for model, color in zip(MODEL_ORDER, palette):
        row = norm.loc[model]
        values = [row[m] for m in METRICS]
        values += values[:1]
        label = MODEL_LABELS[model]
        ax.plot(angles, values, linewidth=2, color=color, label=label)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=16, fontweight="bold")
    ax.tick_params(axis="x", pad=18)
    ax.set_ylim(0, 1.12)

    # Tick labels show the ORIGINAL (unnormalized) values at each axis's best
    # (rim) and worst (near-center) point, matching the existing Downloads
    # PNGs' presentation. Only two labels per axis, not three -- a label
    # placed near radius 0 loses its angular separation from every other
    # axis's near-center label (they all collapse toward the same point),
    # so a middle-ring label is dropped rather than left to collide there.
    for m, angle in zip(METRICS, angles[:-1]):
        raw = norm[f"{m}_raw"]
        lo, hi = raw.min(), raw.max()
        # INVERT metrics (SAM) place their best (lowest raw) value at
        # normalized 1.0, i.e. nearest the rim -- so the rim tick must show
        # the metric's WORST raw value there and vice versa.
        rim_val, center_val = (lo, hi) if INVERT[m] else (hi, lo)
        fmt = (lambda v: f"{v:.1f}") if m == "psnr" else (lambda v: f"{v:.3g}")
        ax.text(angle, 0.98, fmt(rim_val), fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
        ax.text(angle, 0.16, fmt(center_val), fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

    ax.set_yticklabels([])
    ax.set_title(f"{dataset} — Model Performance", fontsize=22, fontweight="bold", pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.08), fontsize=12, frameon=True)
    fig.text(0.5, 0.02,
              "Each metric is min–max normalized within this dataset for radar geometry "
              "(seed-averaged over 67, 69); tick labels show the original values. "
              "SAM is inverted (lower is better).",
              ha="center", fontsize=10, wrap=True)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(REPO_ROOT / "results" / "ablation_table.csv"),
                    help="Path to ablation_table.csv (default: results/ablation_table.csv)")
    ap.add_argument("--out-dir", default=str(SCRIPT_DIR))
    args = ap.parse_args()

    sns.set_theme(style="whitegrid", context="talk")
    input_path = Path(args.input)
    grouped = load_seed_averaged(input_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in DATASET_ORDER:
        if dataset not in grouped["dataset"].unique():
            print(f"SKIP {dataset}: no rows in {input_path}")
            continue
        norm = normalize_for_radar(grouped, dataset)
        plot_dataset(norm, dataset, out_dir / f"{dataset.lower()}_radar.png")


if __name__ == "__main__":
    main()
