"""
visualisations/iclr/make_extended_radar_charts.py
----------------------------------------------------
Seed-averaged 6-axis radar ("spider") charts, one per dataset, for the
extended reconstruction metric pool: SAM, PSNR, SSIM, SID, SCC, Q2^n.
Normalized 0-1 within each dataset for the radar geometry; lower-is-better
metrics (SAM, SID) are inverted so "further out on every axis" always means
"better", matching visualisations/iclr/make_radar_charts.py's convention.

2026-09-19: PRISM (vae-our-nl) is EXCLUDED from this run. scripts/inference.sh
never evaluated vae-our-nl in its reconstruction sweep (fixed in that script
as of this commit, but not yet re-run), so its ablation_table.csv rows are
stale -- frozen at whatever inference.py last wrote before sid/scc/q2n
existed. Re-run scripts/inference.sh on the lab box, then re-run this script
with the fresh ablation_table.csv, to add PRISM's line.

Datasets: IIRS, CRIMS, AVIRIS only (M3 excluded per this run's request).

Usage
=====
    PYTHONPATH=. python visualisations/iclr/make_extended_radar_charts.py \\
        --input "/path/to/ablation_table.csv"
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

# Registry key -> legend label, matching make_radar_charts.py's convention.
# vae-our-nl deliberately absent -- see module docstring.
MODEL_LABELS = {
    "vae-1d-pixelwise": "VAE 1D Pixelwise",
    "vae-3d-spatio-spectral": "VAE 3D Spatio-Spectral",
    "vae-our": "VAE Ours",
    "vae-standard": "VAE Standard",
}
MODEL_ORDER = ["vae-1d-pixelwise", "vae-3d-spatio-spectral", "vae-our", "vae-standard"]

DATASET_ORDER = ["IIRS", "AVIRIS", "CRIMS"]

# Grouped conceptually: spectral-fidelity metrics (SAM, SID) adjacent, then
# spatial/structural metrics (PSNR, SSIM, SCC), then the combined hypercomplex
# index (Q2^n) last.
METRICS = ["sam_valid", "sid", "psnr", "ssim", "scc", "q2n"]
METRIC_LABELS = {"sam_valid": "SAM", "sid": "SID", "psnr": "PSNR",
                  "ssim": "SSIM", "scc": "SCC", "q2n": "Q2ⁿ"}
# Lower-is-better metrics get inverted during normalization so "further out
# on the chart" means "better" on every axis, matching make_radar_charts.py.
INVERT = {"sam_valid": True, "sid": True, "psnr": False, "ssim": False,
          "scc": False, "q2n": False}


def load_seed_averaged(input_path: Path) -> pd.DataFrame:
    """
    Filter to the SAM-selected checkpoint and the physics loss regime, then
    average the six metrics across seeds.

    Same two filters as make_radar_charts.py, for the same reasons:
      - select == "sam": avoids M3-style select=mse duplicate rows silently
        pooling some models over more rows than others (not currently present
        for these 3 datasets/4 models, but kept as a guard).
      - loss == "physics": vae-our is physics-only; the three baselines have
        both regimes -- physics-loss rows are used for all four models for a
        like-for-like comparison (the grid's own framing of the physics-loss
        comparison as "the claim" being tested).
    """
    df = pd.read_csv(input_path)
    df = df[(df["select"] == "sam") & (df["loss"] == "physics")
            & (df["dataset"].isin(DATASET_ORDER)) & (df["model"].isin(MODEL_ORDER))]
    missing_models = set(MODEL_ORDER) - set(df["model"].unique())
    if missing_models:
        raise SystemExit(f"No physics/select=sam rows for: {sorted(missing_models)} "
                          f"in {input_path} -- check the input file.")
    empty_metric_cells = df[METRICS].isna().all(axis=0)
    if empty_metric_cells.any():
        raise SystemExit(f"Entirely empty metric column(s): "
                          f"{list(empty_metric_cells[empty_metric_cells].index)} "
                          f"in {input_path} -- these rows likely predate that "
                          f"metric being computed. Re-run scripts/inference.sh.")
    grouped = (df.groupby(["dataset", "model"])[METRICS]
                 .mean()
                 .reset_index())
    counts = df.groupby(["dataset", "model"]).size()
    off = counts[counts != 2]
    if len(off):
        print(f"NOTE: non-2-seed groups (averaged anyway, but check these):\n{off}")
    missing = grouped[grouped[METRICS].isna().any(axis=1)]
    if len(missing):
        print(f"WARNING: some (dataset, model) cells are missing one or more "
              f"metrics and will render as gaps in the chart:\n{missing}")
    return grouped


def normalize_for_radar(grouped: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Per-dataset min-max normalize each metric to [0, 1]; invert SAM/SID."""
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

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    palette = sns.color_palette("tab10", n_colors=len(MODEL_ORDER))

    for model, color in zip(MODEL_ORDER, palette):
        row = norm.loc[model]
        values = [row[m] for m in METRICS]
        values += values[:1]
        label = MODEL_LABELS[model]
        ax.plot(angles, values, linewidth=2, color=color, label=label)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=15, fontweight="bold")
    ax.tick_params(axis="x", pad=22)
    ax.set_ylim(0, 1.12)

    # Tick labels show the ORIGINAL (unnormalized) values at each axis's best
    # (rim) and worst (near-center) point -- only two per axis, not three: a
    # label placed near radius 0 loses its angular separation from every
    # other axis's near-center label (they all collapse toward the same
    # point with 6 axes even more than with 3), so a middle-ring label is
    # skipped rather than left to collide there.
    for m, angle in zip(METRICS, angles[:-1]):
        raw = norm[f"{m}_raw"]
        lo, hi = raw.min(), raw.max()
        rim_val, center_val = (lo, hi) if INVERT[m] else (hi, lo)
        fmt = (lambda v: f"{v:.1f}") if m == "psnr" else (lambda v: f"{v:.3g}")
        ax.text(angle, 0.98, fmt(rim_val), fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))
        ax.text(angle, 0.16, fmt(center_val), fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))

    ax.set_yticklabels([])
    ax.set_title(f"{dataset} — Extended Metric Pool", fontsize=22, fontweight="bold", pad=36)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.08), fontsize=12, frameon=True)
    fig.text(0.5, 0.02,
              "Each metric is min–max normalized within this dataset for radar geometry "
              "(seed-averaged over 67, 69, physics loss); tick labels show the original "
              "values. SAM and SID are inverted (lower is better). Q2ⁿ is comparable "
              "within this dataset only (band-padding fraction differs by dataset).",
              ha="center", fontsize=9.5, wrap=True)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help="Path to the ablation_table.csv to read (no default -- "
                         "point this at the freshest available CSV explicitly).")
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
        plot_dataset(norm, dataset, out_dir / f"extended_metrics_{dataset.lower()}_radar.png")


if __name__ == "__main__":
    main()
