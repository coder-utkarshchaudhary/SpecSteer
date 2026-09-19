#!/usr/bin/env python3
"""
paper/iclr/v2-claude/make_tables.py
------------------------------------
Reads results/ablation_table.csv, results/downstream_table.csv,
results/probes.csv, and results/model_params.json, and emits LaTeX table-BODY
fragments (booktabs rows only — no \\begin{tabular}, no header, no \\caption)
under paper/iclr/v2-claude/tables/, so the paper's own .tex keeps the
preamble/footer stable and just \\input{}s these as the numbers refresh.

Run after every results/ refresh:
    PYTHONPATH=. python paper/iclr/v2-claude/make_tables.py

Filters applied everywhere: dataset in {IIRS, AVIRIS, CRIMS} (M3 excluded per
the paper's scope decision), model in the four the paper actually compares
(vae-our is dropped — PRISM = vae-our-nl only), loss == physics (the paper's
comparison arm), select == sam (val-SAM-selected checkpoints, stated once in
the experimental setup), seed-averaged over {67, 69}.

A cell that is NaN in the source CSV (e.g. vae-our-nl's SID/SCC/Q2n before the
2026-09-19 rerun backfills them) renders as `\\tbd{}` rather than crashing —
this script is meant to run BOTH tonight (partial data) and after tomorrow's
rerun (complete data) without any code change.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

RAD2DEG = 180.0 / math.pi

DATASET_ORDER = ["IIRS", "AVIRIS", "CRIMS"]
# The CSVs / codebase spell it CRIMS throughout; the instrument's own name and
# the paper's prose use CRISM. Display-only remap, data stays keyed on CRIMS.
DATASET_DISPLAY = {"IIRS": "IIRS", "AVIRIS": "AVIRIS", "CRIMS": "CRISM"}

MODEL_ORDER = ["vae-standard", "vae-1d-pixelwise", "vae-3d-spatio-spectral", "vae-our-nl"]
MODEL_DISPLAY = {
    "vae-standard": "2D Spatial VAE",
    "vae-1d-pixelwise": "1D Pixelwise VAE",
    "vae-3d-spatio-spectral": "3D Spatio-Spectral VAE",
    "vae-our-nl": r"\textbf{PRISM}",
}
SEEDS = {67, 69}
LOSS = "physics"
SELECT = "sam"
TBD = r"\tbd{}"


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        print(f"WARNING: {path} does not exist — treating as empty.", file=sys.stderr)
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def to_float(s) -> float:
    if s is None or s == "" or s == "nan":
        return float("nan")
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def filter_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r.get("dataset") not in DATASET_ORDER:
            continue
        if r.get("model") not in MODEL_ORDER:
            continue
        if r.get("loss") != LOSS:
            continue
        if r.get("select") != SELECT:
            continue
        try:
            if int(r.get("seed")) not in SEEDS:
                continue
        except (TypeError, ValueError):
            continue
        out.append(r)
    return out


def seed_mean(rows: list[dict], keys: list[str]) -> dict[tuple[str, str], dict]:
    """{(dataset, model): {key: nanmean over seeds, "_n_seeds": n contributing}}"""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["model"]), []).append(r)

    out = {}
    for gk, grs in groups.items():
        entry = {}
        for k in keys:
            vals = [to_float(r.get(k)) for r in grs]
            valid = [v for v in vals if not math.isnan(v)]
            entry[k] = float(np.mean(valid)) if valid else float("nan")
        entry["_n_seeds"] = sum(1 for r in grs if not math.isnan(to_float(r.get(keys[0]))))
        out[gk] = entry
        if len(grs) < len(SEEDS):
            print(f"WARN {gk[0]}/{gk[1]}: {len(grs)} seed row(s) found, expected "
                  f"{len(SEEDS)} — presented as a mean over what's available.",
                  file=sys.stderr)
    return out


def load_params(path: Path) -> dict[tuple[str, str], float]:
    if not path.is_file():
        print(f"WARNING: {path} does not exist — Params column will be {TBD}. "
              f"Run: PYTHONPATH=. python utils/check-model-params.py --json {path}",
              file=sys.stderr)
        return {}
    data = json.loads(path.read_text())
    out = {}
    for ds, models in data.items():
        for model, n in models.items():
            out[(ds, model)] = n / 1e6
    return out


@dataclass
class Col:
    key: str
    header: str
    prec: int
    hib: bool | None          # True = higher is better (bold max), False = lower is better
                              # (bold min), None = context column, never bolded
    transform: object = None  # callable applied to the raw value before formatting
    fmt: str = "f"            # "f" | "d"


def get_value(table: dict, ds: str, model: str, col: Col, params: dict) -> float:
    if col.key == "params_m":
        v = params.get((ds, model), float("nan"))
    else:
        v = table.get((ds, model), {}).get(col.key, float("nan"))
    if col.transform is not None and not math.isnan(v):
        v = col.transform(v)
    return v


def best_cells(table: dict, params: dict, col: Col) -> set[tuple[str, str]]:
    if col.hib is None:
        return set()
    best = set()
    for ds in DATASET_ORDER:
        vals = {m: get_value(table, ds, m, col, params) for m in MODEL_ORDER}
        finite = {m: v for m, v in vals.items() if not math.isnan(v)}
        if not finite:
            continue
        pick = max(finite, key=finite.get) if col.hib else min(finite, key=finite.get)
        best.add((ds, pick))
    return best


def fmt_cell(v: float, col: Col, is_best: bool) -> str:
    if math.isnan(v):
        return TBD
    if col.fmt == "d":
        s = f"{v:.0f}"
    else:
        s = f"{v:.{col.prec}f}"
    return rf"\textbf{{{s}}}" if is_best else s


def render_body(table: dict, params: dict, cols: list[Col]) -> str:
    """
    Row bodies AND the closing \\bottomrule (see the note below) — no
    \\begin{tabular}, header row, or \\caption; the paper's own .tex keeps
    those and \\input{}s this.

    \\bottomrule is emitted HERE, as the fragment's own last line, rather
    than left in the outer .tex immediately after \\input{...}: booktabs'
    \\toprule/\\midrule/\\bottomrule peek at the token following a row's `\\`
    to merge inter-row spacing, and that lookahead breaks (a "Misplaced
    \\noalign" / "Undefined control sequence \\@BTrule" cascade) when the
    token it needs to see is on the OTHER side of an \\input file boundary.
    Ending each fragment with its own \\bottomrule keeps the lookahead
    entirely inside one file. Confirmed empirically: \\input followed by an
    ordinary row works, \\input followed immediately by \\hline/\\bottomrule
    does not, and moving \\bottomrule inside the \\input'd file fixes it.
    """
    lines = []
    n_models = len(MODEL_ORDER)
    best = {col.key: best_cells(table, params, col) for col in cols}
    for di, ds in enumerate(DATASET_ORDER):
        for mi, model in enumerate(MODEL_ORDER):
            cells = []
            for col in cols:
                v = get_value(table, ds, model, col, params)
                is_best = (ds, model) in best[col.key]
                cells.append(fmt_cell(v, col, is_best))
            row_cells = " & ".join(cells)
            if mi == 0:
                prefix = rf"\multirow{{{n_models}}}{{*}}{{{DATASET_DISPLAY[ds]}}}"
            else:
                prefix = ""
            model_name = MODEL_DISPLAY[model]
            lines.append(f"{prefix} & {model_name} & {row_cells} \\\\")
        if di < len(DATASET_ORDER) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    return "\n".join(lines) + "\n"


def write(path: Path, body: str, provenance: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "% GENERATED by paper/iclr/v2-claude/make_tables.py — DO NOT EDIT BY HAND\n"
        f"% {provenance}\n"
    )
    path.write_text(header + body)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Table specs
# ---------------------------------------------------------------------------

TABLE1 = [
    Col("sam_valid", r"SAM-valid ($^\circ$) $\downarrow$", 2, False, lambda v: v * RAD2DEG),
    Col("psnr", r"PSNR (dB) $\uparrow$", 2, True),
    Col("ssim", r"SSIM $\uparrow$", 3, True),
    Col("sid", r"SID $\downarrow$", 4, False),
    Col("scc", r"SCC $\uparrow$", 3, True),
    Col("q2n", r"Q$2^n$ $\uparrow$", 3, True),
    Col("params_m", r"Params (M)", 2, None),
]

TABLE2 = [
    Col("psnr_mid", r"PSNR@$\sigma{=}0.5$ (dB) $\uparrow$", 2, True),
    Col("sam_mid", r"SAM@$\sigma{=}0.5$ ($^\circ$) $\downarrow$", 2, False, lambda v: v * RAD2DEG),
    Col("psnr_drop", r"PSNR drop (dB)", 2, None),
]

TABLE3 = [
    Col("jaggedness", r"Jaggedness $\downarrow$", 4, False),
    Col("path_length", r"Path length", 3, None),
]

TABLE4 = [
    Col("p5_mask_bands", r"Masked bands", 0, None, fmt="d"),
    Col("p5_masked_psnr", r"PSNR (masked) $\uparrow$", 2, True),
    Col("p5_masked_psnr_meanfill", r"Mean-fill PSNR", 2, None),
    Col("p5_masked_sam_valid", r"SAM-valid ($^\circ$) $\downarrow$", 2, False, lambda v: v * RAD2DEG),
    Col("p5_relative_gain", r"Rel.\ gain $\uparrow$", 3, True),
    Col("p5_passthrough_index", r"Pass-through", 3, None),
]

TABLE5 = [
    Col("p1_floor_psnr", r"Floor PSNR", 2, None),
    Col("p1_lift_psnr_db", r"Lift (dB) $\uparrow$", 2, True),
    Col("p1_lift_sam_valid_rel", r"Lift SAM-v (rel.) $\uparrow$", 3, True),
    Col("p1_lift_vs_meanpatch_psnr_db", r"vs.\ mean-patch (dB)", 2, None),
    Col("p1_headroom_psnr", r"Headroom $\uparrow$", 3, True),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--out-dir", default=str(SCRIPT_DIR / "tables"))
    ap.add_argument("--params-json", default=None)
    ap.add_argument("--tables", nargs="*", default=None,
                    choices=["1", "2", "3", "4", "5"],
                    help="Subset of tables to (re)generate. Default: all.")
    ap.add_argument("--strict", action="store_true",
                    help="Exit 1 if any rendered cell is TBD (use once the "
                         "vae-our-nl rerun is complete).")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    params_json = Path(args.params_json) if args.params_json else results_dir / "model_params.json"
    want = set(args.tables) if args.tables else {"1", "2", "3", "4", "5"}

    ablation = filter_rows(read_csv(results_dir / "ablation_table.csv"))
    downstream = filter_rows(read_csv(results_dir / "downstream_table.csv"))
    probes = filter_rows(read_csv(results_dir / "probes.csv"))
    params = load_params(params_json)

    any_tbd = False

    if "1" in want:
        t1 = seed_mean(ablation, ["sam_valid", "psnr", "ssim", "sid", "scc", "q2n"])
        body = render_body(t1, params, TABLE1)
        any_tbd |= TBD in body
        write(out_dir / "table1_reconstruction.tex", body,
              f"source: {results_dir}/ablation_table.csv + {params_json} | "
              f"seeds={sorted(SEEDS)} loss={LOSS} select={SELECT}")

    if "2" in want:
        t2 = seed_mean(downstream, ["psnr_mid", "sam_mid", "psnr_drop"])
        body = render_body(t2, {}, TABLE2)
        any_tbd |= TBD in body
        write(out_dir / "table2_noise.tex", body,
              f"source: {results_dir}/downstream_table.csv | seeds={sorted(SEEDS)} "
              f"loss={LOSS} select={SELECT} | sigma=0.5 only (psnr_clean/psnr_mid=sigma0/sigma0.5)")

    if "3" in want:
        t3 = seed_mean(downstream, ["jaggedness", "path_length"])
        body = render_body(t3, {}, TABLE3)
        any_tbd |= TBD in body
        write(out_dir / "table3_interpolation.tex", body,
              f"source: {results_dir}/downstream_table.csv | seeds={sorted(SEEDS)} "
              f"loss={LOSS} select={SELECT}")

    if "4" in want:
        t4 = seed_mean(probes, ["p5_mask_bands", "p5_masked_psnr", "p5_masked_psnr_meanfill",
                                "p5_masked_sam_valid", "p5_relative_gain", "p5_passthrough_index"])
        body = render_body(t4, {}, TABLE4)
        any_tbd |= TBD in body
        write(out_dir / "table4_inpainting.tex", body,
              f"source: {results_dir}/probes.csv (P5 band-masking) | seeds={sorted(SEEDS)} "
              f"loss={LOSS} select={SELECT}")

    if "5" in want:
        t5 = seed_mean(probes, ["p1_floor_psnr", "p1_lift_psnr_db", "p1_lift_sam_valid_rel",
                                "p1_lift_vs_meanpatch_psnr_db", "p1_headroom_psnr"])
        body = render_body(t5, {}, TABLE5)
        any_tbd |= TBD in body
        write(out_dir / "table5_floors.tex", body,
              f"source: {results_dir}/probes.csv (P1 trivial floors, diagnostic only) | "
              f"seeds={sorted(SEEDS)} loss={LOSS} select={SELECT}")

    if args.strict and any_tbd:
        print("STRICT: at least one cell is TBD.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
