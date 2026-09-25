"""
inference/paper_tables_final.py
-------------------------------
CSV -> paste-ready LaTeX for the paper's Table 4 (input-space noise recovery)
and Table 6 (missing-pixel recovery), plus a plain-text win/loss summary.

Inputs (all optional — a missing file just leaves its columns as "--"):
    results/final/noise-recovery.csv                 sigma 0.01, 0.05, 0.2
    results/final_sigma01/noise-recovery.csv         sigma 0.0 (clean), 0.1
    results/final/missing_pixels/missing-pixel-recovery.csv

Only measured numbers are printed. Physics-objective rows only for the
baselines (the paper's Table 3/4 model list); PRISM = vae-our-nl.

    PYTHONPATH=. python inference/paper_tables_final.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

MODELS = [  # (registry name, display name) in the paper's row order
    ("vae-standard", "2D Spatial VAE"),
    ("vae-1d-pixelwise", "1D Pixelwise VAE"),
    ("vae-3d-spatio-spectral", "3D Spatio-Spectral VAE"),
    ("vae-our-nl", "PRISM"),
]
DATASETS = [("IIRS", "IIRS"), ("AVIRIS", "AVIRIS"), ("CRIMS", "CRISM")]
TABLE4_SIGMAS = ["0.01", "0.05", "0.1", "0.2"]
NOISY_REF = "noisy_input_no_model"
DEG = 180.0 / math.pi


def _read(path: Path) -> pd.DataFrame | None:
    if path.is_file():
        return pd.read_csv(path)
    print(f"  (missing, columns will be '--'): {path}")
    return None


def _merge_noise(paths: list[Path]) -> pd.DataFrame | None:
    frames = [f for f in (_read(p) for p in paths) if f is not None]
    if not frames:
        return None
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["dataset", "model", "loss"], how="outer")
    return out


def _pick(df: pd.DataFrame | None, ds: str, model: str) -> pd.Series | None:
    if df is None:
        return None
    loss = "none" if model == NOISY_REF else "physics"
    rows = df[(df.dataset == ds) & (df.model == model) & (df.loss == loss)]
    return rows.iloc[0] if len(rows) else None


def _val(row: pd.Series | None, col: str, scale: float = 1.0) -> float:
    if row is None or col not in row or pd.isna(row[col]):
        return float("nan")
    return float(row[col]) * scale


def _fmt(v: float, nd: int, bold: bool) -> str:
    if math.isnan(v):
        return "--"
    s = f"{v:.{nd}f}"
    return f"\\textbf{{{s}}}" if bold else s


def _best(vals: list[float], higher: bool) -> float:
    ok = [v for v in vals if not math.isnan(v)]
    return (max(ok) if higher else min(ok)) if ok else float("nan")


def build_table(df, columns, caption: str, label: str, extra_rows=None, header=None):
    """columns: list of (header, csv_col, scale, decimals, higher_is_better|None)."""
    summary = []
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{ll" + "c" * len(columns) + "}", "\\toprule",
    ] + (header or ["Dataset & Model & " + " & ".join(h for h, *_ in columns) + " \\\\"]) + [
        "\\midrule",
    ]
    for ds, ds_label in DATASETS:
        vals = {m: [_val(_pick(df, ds, m), c, sc) for _, c, sc, _, _ in columns] for m, _ in MODELS}
        bests = [_best([vals[m][j] for m, _ in MODELS], hb) if hb is not None else None
                 for j, (*_, hb) in enumerate(columns)]
        block = []
        for i, (m, name) in enumerate(MODELS):
            cells = [_fmt(v, nd, b is not None and not math.isnan(v) and abs(v - b) < 1e-12)
                     for v, b, (_, _, _, nd, _) in zip(vals[m], bests, columns)]
            block.append(f"{ds_label if i == 0 else ''} & {name} & " + " & ".join(cells) + " \\\\")
        for er in (extra_rows or []):
            row = _pick(df, ds, er["model"])
            cells = [_fmt(_val(row, c, sc), nd, False) if c in er["cols"] else "--"
                     for _, c, sc, nd, _ in columns]
            block.append(f" & \\textit{{{er['label']}}} & " + " & ".join(cells) + " \\\\")
        lines += block + ["\\midrule"]
        for j, (h, *_rest) in enumerate(columns):
            hb = columns[j][4]
            if hb is None:
                continue
            ranked = sorted(((vals[m][j], name) for m, name in MODELS if not math.isnan(vals[m][j])),
                            reverse=hb)
            if not ranked:
                continue
            prism = vals["vae-our-nl"][j]
            rank = next((k + 1 for k, (_, n) in enumerate(ranked) if n == "PRISM"), None)
            summary.append(f"{ds_label:7} {h:28} best={ranked[0][1]} ({ranked[0][0]:.4g})  "
                           f"PRISM={prism:.4g} rank={rank}/{len(ranked)}"
                           + ("  <-- WIN" if rank == 1 else ""))
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(lines) + "\n", summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--noise", nargs="+", type=Path,
                   default=[Path("results/final/noise-recovery.csv"),
                            Path("results/final_sigma01/noise-recovery.csv")])
    p.add_argument("--missing-pixels", type=Path,
                   default=Path("results/final/missing_pixels/missing-pixel-recovery.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("results/final/paper_snippets"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    noise = _merge_noise(args.noise)
    if noise is not None and {"psnr_recovery_s0.0", "psnr_recovery_s0.2"} <= set(noise.columns):
        # like-for-like drop: both are means of per-sample PSNR from the same pipeline
        noise["psnr_drop_s0.2"] = noise["psnr_recovery_s0.0"] - noise["psnr_recovery_s0.2"]
    t4_cols = []
    for s in TABLE4_SIGMAS:
        t4_cols += [(f"PSNR sigma={s}", f"psnr_recovery_s{s}", 1.0, 2, True),
                    (f"SAM-v sigma={s}", f"sam_recovery_s{s}", DEG, 2, False)]
    t4_cols.append(("dPSNR@0.2", "psnr_drop_s0.2", 1.0, 2, None))
    t4_header = [
        " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$\\sigma$={s}}}" for s in TABLE4_SIGMAS)
        + " & \\\\",
        " ".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(TABLE4_SIGMAS))),
        "Dataset & Model & " + " & ".join(["PSNR$\\uparrow$ & SAM-v ($^\\circ$)$\\downarrow$"] * len(TABLE4_SIGMAS))
        + " & $\\Delta$PSNR@0.2 \\\\",
    ]
    t4, s4 = build_table(
        noise, t4_cols,
        caption=("Recovery from input-space corruption. Gaussian noise of standard deviation "
                 "$\\sigma$ times each band's own signal standard deviation is added to the "
                 "input cube, which is then encoded and decoded; the output is scored against "
                 "the clean cube. PSNR is the mean of per-patch PSNR; SAM-valid excludes "
                 "pixels below the SAM energy floor. Averaged over two trained checkpoints "
                 "(seeds 67, 69) and three noise draws. \\emph{Noisy input} scores the corrupted "
                 "cube itself, with no model: a model whose PSNR exceeds it removes noise rather "
                 "than passing it through. $\\Delta$PSNR@0.2 is the drop from the same model's "
                 "clean-input reconstruction, computed in the same per-patch convention."),
        label="tab:noise",
        header=t4_header,
        extra_rows=[{"model": NOISY_REF, "label": "Noisy input (no model)",
                     "cols": {c for _, c, *_ in t4_cols if c != "psnr_drop_s0.2"}}],
    )

    mp = _read(args.missing_pixels)
    t6_cols = [
        ("PSNR clean$\\uparrow$", "psnr_clean", 1.0, 2, True),
        ("PSNR masked$\\uparrow$", "psnr_masked", 1.0, 2, True),
        ("$\\Delta$PSNR (dB)$\\downarrow$", "psnr_drop", 1.0, 2, False),
        ("SAM-v masked ($^\\circ$)$\\downarrow$", "sam_valid_masked", DEG, 2, False),
        ("$\\Delta$SAM-v ($^\\circ$)$\\downarrow$", "sam_valid_drop", DEG, 2, False),
    ]
    t6, s6 = build_table(
        mp, t6_cols,
        caption=("Missing-pixel recovery. A random 10\\% of each patch's pixels are zeroed "
                 "across all bands at the model's input; the reconstruction is scored against "
                 "the clean cube, whole-cube, next to the same model's reconstruction of the "
                 "unmasked cube. PSNR from the pooled MSE, as in Table~3. One checkpoint "
                 "(seed 67), one mask draw."),
        label="tab:missing_pixels",
    )

    (args.out_dir / "table4_noise.tex").write_text(t4)
    (args.out_dir / "table6_missing_pixels.tex").write_text(t6)
    summary = ["== Table 4 (input noise) =="] + s4 + ["", "== Table 6 (missing pixels) =="] + s6
    (args.out_dir / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))
    print(f"\nwrote {args.out_dir}/table4_noise.tex, table6_missing_pixels.tex, summary.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
