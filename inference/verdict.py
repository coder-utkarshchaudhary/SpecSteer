"""
inference/verdict.py
--------------------
Aggregate the per-cell diagnostic JSONs (inference/probes.py) into three
analysis artifacts:

    results/probes.csv       one row per cell: reconstruction metrics + the
                             P2/P3/P4 diagnostics
    results/stats.csv        pairwise model comparisons with bootstrap CIs,
                             permutation p-values (Holm-corrected), effect sizes
    results/DIAGNOSTICS.txt  the human-readable summary

2026-09-04 DEMOTION (docs/new_plan.md): this module used to render VERDICT.txt
with a suite-level PASS/FAIL/INVALID adjudication driven by the P1 trivial
floor. That floor was miscalibrated (it flagged every cell INVALID, including
a 39 dB vae-3d), so the adjudication layer was retired together with probes
P1/P5/P6/P7. What remains has no pass/fail semantics:

  * the latent-rate audit (P2)      — the fairness certificate,
  * collapse detection (P3)         — the one exclusion that survives: a
                                      collapsed cell is skipped by the stats,
  * spatial reliance / SRI (P4)     — the architecture-story figure,
  * sam_valid                       — the cross-dataset-comparable SAM,
  * the paired statistics           — rankings still need significance + the
                                      preregistered effect floors; they just no
                                      longer pass through a verdict gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from inference.stats import apply_holm, compare  # noqa: E402

# Metric orientation: which direction counts as better, and the preregistered
# minimum difference that counts as meaningful at all.
METRICS = {
    "psnr": ("psnr_db", True),
    "ssim": ("ssim", True),
    "sam": ("sam_rad", False),
    "sam_valid": ("sam_rad", False),
}


def load_cells(d: Path) -> list[dict]:
    out = []
    for f in sorted(d.glob("*.json")):
        if f.name.startswith("floors_"):
            continue
        try:
            out.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            print(f"  ! unreadable: {f}")
    return [c for c in out if not c.get("error")]


def flatten(c: dict) -> dict:
    p2, p3, p4 = c["P2_latent_budget"], c["P3_collapse"], c["P4_spatial_reliance"]
    r = c["reconstruction"]
    row = {
        "dataset": c["dataset"], "model": c["model"], "loss": c["loss"],
        "seed": c.get("seed"), "select": c.get("select"),
        "n_patches": c["n_patches"], "epochs": c.get("trained_epochs"),
        "mse": r["mse"], "psnr": r["psnr"], "ssim": r["ssim"],
        "sam": r["sam"], "sam_valid": r["sam_valid"],
        # P2 — rate audit
        "latent_elements": p2["latent_elements"], "compression_ratio": p2["compression_ratio"],
        "rate_dev_pct": p2["deviation_pct"], "rate_matched": p2["rate_matched"],
        # P3 — collapse
        "active_units": p3["active_unit_fraction"], "latent_swap_delta": p3["latent_swap_delta"],
        "collapsed": c.get("collapsed", p3["collapsed"]),
        # P4 — spatial reliance
        "sri": p4["sri"], "sri_note": p4.get("note", ""),
    }
    if "per_branch_mse" in p2:
        row.update({f"our_{k}": v for k, v in p2["per_branch_mse"].items()})
    return row


def _cell_label(c: dict) -> str:
    """`model|loss|seed<N>` — the seed keeps the Holm family from containing a
    cell compared against itself once the grid has several seeds per cell."""
    s = c.get("seed")
    return f"{c['model']}|{c['loss']}" + (f"|seed{s}" if s is not None else "")


def pairwise(cells: list[dict], cfg: dict) -> list[dict]:
    """
    Every model pair within a (dataset, seed), on the shared patch sample.

    Grouping by seed as well as dataset: the Holm family is the set of tests you
    look across before making one claim, and that claim is "model A beats model
    B at a given seed". Mixing seeds into one family would both pad it with
    same-model/different-seed pairs (a nondeterminism measurement, not a model
    comparison) and, before the seed made it into the label, compare a cell
    against itself.

    Collapsed cells are excluded: a collapsed decoder emits a constant, so its
    per-patch metrics describe a trivial predictor, not a model.
    """
    st = cfg["p8_statistics"]
    rows = []
    by_grp: dict[tuple, list[dict]] = {}
    for c in cells:
        by_grp.setdefault((c["dataset"], c.get("seed")), []).append(c)

    for (ds, seed), cs in sorted(by_grp.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        usable = [c for c in cs
                  if not c.get("collapsed", c["P3_collapse"]["collapsed"])]
        for metric, (eff_key, hib) in METRICS.items():
            min_eff = st["min_meaningful_effect"][eff_key]
            family = []
            for i in range(len(usable)):
                for j in range(i + 1, len(usable)):
                    a, b = usable[i], usable[j]
                    va = np.asarray(a["per_patch"][metric], dtype=np.float64)
                    vb = np.asarray(b["per_patch"][metric], dtype=np.float64)
                    n = min(va.size, vb.size)
                    if n < 4:
                        continue
                    family.append(compare(
                        metric, _cell_label(a), _cell_label(b),
                        va[:n], vb[:n], min_effect=min_eff, higher_is_better=hib,
                        cfg=st, seed=cfg["sampling"]["seed"]))
            for r in apply_holm(family, st):
                d = r.as_row(); d["dataset"] = ds; d["seed"] = seed
                rows.append(d)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Still create the file — callers and the pipeline treat it as a
        # guaranteed artifact. An empty stats.csv is the correct output when
        # every cell is collapsed (nothing to compare).
        path.write_text("")
        return
    cols = list({k: None for r in rows for k in r})
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def render_diagnostics(rows: list[dict], stats_rows: list[dict], cfg: dict) -> str:
    L = []
    A = L.append
    A("=" * 78)
    A(" MECHANISM DIAGNOSTICS")
    A(f" preregistered {cfg.get('registered_on')} · split {cfg.get('split')} "
      f"· {cfg['sampling']['max_patches']} patches/cell")
    A("=" * 78)
    A("")
    A("Diagnostics: P2 latent rate · P3 collapse · P4 spatial reliance · SAMv.")
    A("No pass/fail adjudication (retired 2026-09-04). The one exclusion is")
    A("collapse: a collapsed decoder emits a constant, so its metrics describe")
    A("a trivial predictor — such cells are findings, not scores.")

    by_ds: dict[str, list[dict]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)

    def lbl(r):
        s = r.get("seed")
        return f"{r['model']}|{r['loss']}" + (f"|s{s}" if s is not None else "")

    def sort_key(r):
        return (r["model"], r["loss"], r.get("seed") or 0)

    for ds, rs in sorted(by_ds.items()):
        A("")
        A("=" * 78)
        A(f" {ds}")
        A("=" * 78)
        A("")
        A(f"  {'model|loss|seed':<36}{'PSNR':>7}{'SSIM':>7}{'SAMv':>8}  status")
        A("  " + "-" * 70)
        for r in sorted(rs, key=lambda z: (-z["psnr"], sort_key(z))):
            status = "COLLAPSED — excluded from stats" if r["collapsed"] else ""
            A(f"  {lbl(r):<36}{r['psnr']:>7.2f}{r['ssim']:>7.3f}"
              f"{r['sam_valid']:>8.4f}  {status}")

        A("")
        A("  Latent rate (must be matched for the reconstruction comparison to mean anything):")
        for r in sorted(rs, key=sort_key):
            ok = "matched" if r["rate_matched"] else "UNMATCHED"
            A(f"    {lbl(r):<36}{r['latent_elements']:>9,} elements"
              f"  {r['compression_ratio']:>7.1f}:1  ({r['rate_dev_pct']:+5.1f}%)  {ok}")

        A("")
        A("  Mechanism:")
        A(f"    {'model|loss|seed':<36}{'SRI':>8}{'active':>8}{'swapDSAM':>10}")
        for r in sorted(rs, key=sort_key):
            A(f"    {lbl(r):<36}{r['sri']:>8.3f}"
              f"{r['active_units']:>8.2f}{r['latent_swap_delta']:>10.3f}")
        A("      SRI      < 0.02 => uses no spatial context (expected for vae-1d;")
        A("               the Iteration-1 before/after number for vae-our)")
        A("      active   fraction of latent dims with non-trivial KL")
        A("      swapDSAM relative SAM change when decoding another patch's latent")
        A("               (~0 => the decoder ignores the latent)")

        ours = sorted((r for r in rs if r["model"] == "vae-our" and "our_mse_final" in r),
                      key=sort_key)
        if ours:
            o = ours[0]
            A("")
            A(f"  vae-our loss decomposition ({lbl(o)}; mix 0.5:w:w normalised — "
              f"see modules/vae_our.py):")
            A(f"    mse_final    {o['our_mse_final']:.6f}   <- the fused output; the ONLY")
            A("                              cross-model comparable reconstruction number")
            A(f"    mse_spatial  {o['our_mse_spatial']:.6f}   <- spatial stream alone (8x8 grid latent)")
            A(f"    mse_spectral {o['our_mse_spectral']:.6f}   <- spectral stream alone")
            A(f"    final contributes {100 * o['our_final_share_of_total']:.1f}% of the training mix.")

        srs = [s for s in stats_rows if s["dataset"] == ds and s["metric"] == "sam_valid"]
        if srs:
            A("")
            A("  Pairwise SAM-valid (Holm-corrected, with the preregistered effect floor):")
            for s in sorted(srs, key=lambda z: z["p_holm"] or 1.0):
                A(f"    {s['model_a']:<26} vs {s['model_b']:<26} "
                  f"d={s['delta']:+.4f} [{s['ci_low']:+.4f},{s['ci_high']:+.4f}] "
                  f"p={s['p_holm']:.4f}  {s['verdict']}")

    A("")
    A("=" * 78)
    A(" HOW TO READ THIS")
    A("=" * 78)
    A(" A ranking claim needs: the cell not collapsed, the rate matched, and the")
    A(" pairwise difference both significant after Holm AND above the")
    A(" preregistered effect floor. 'significant_but_negligible' is not a win —")
    A(" with hundreds of paired patches almost any difference reaches p<0.05.")
    A(" Raw SAM is not comparable across datasets (near-zero pixels contribute")
    A(" exactly pi/2 whatever the model predicts); use SAMv.")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate diagnostic results into probes.csv, stats.csv, "
                    "DIAGNOSTICS.txt.")
    ap.add_argument("--probes-dir", type=Path, default=Path("results/probes"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--prereg", type=Path,
                    default=REPO_ROOT / "inference" / "preregistration.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.prereg.read_text())
    cells = load_cells(args.probes_dir)
    if not cells:
        print(f"no diagnostic results under {args.probes_dir}")
        return 1

    rows = [flatten(c) for c in cells]
    stats_rows = pairwise(cells, cfg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "probes.csv")
    write_csv(stats_rows, args.out_dir / "stats.csv")
    txt = render_diagnostics(rows, stats_rows, cfg)
    (args.out_dir / "DIAGNOSTICS.txt").write_text(txt + "\n")
    print(txt)
    print(f"\nwrote {args.out_dir}/probes.csv, stats.csv, DIAGNOSTICS.txt "
          f"({len(rows)} cells, {len(stats_rows)} comparisons)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
