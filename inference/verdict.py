"""
inference/verdict.py
--------------------
Turn the per-cell probe JSONs into the three analysis artifacts:

    results/probes.csv    one row per cell, every probe metric + verdict
    results/stats.csv     pairwise model comparisons with CIs, p, effect sizes
    results/VERDICT.txt   the human-readable answer to "why does my model beat
                          or get beaten on each dataset"

VERDICT.txt is the deliverable. It reports, per dataset: which cells are usable
at all, how much of the achievable headroom each captured, whether the rate
match held, which probes each model passed, and which pairwise differences are
both statistically significant AND large enough to matter.
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
    p1, p2, p3 = c["P1_trivial_floors"], c["P2_latent_budget"], c["P3_collapse"]
    p4, p5, p6, p7 = (c["P4_spatial_reliance"], c["P5_spectral_inpainting"],
                      c["P6_purification"], c["P7_linear_probe"])
    r = c["reconstruction"]
    row = {
        "dataset": c["dataset"], "model": c["model"], "loss": c["loss"],
        "verdict": c["verdict"], "reason": c.get("verdict_reason", ""),
        "n_patches": c["n_patches"], "epochs": c.get("trained_epochs"),
        "mse": r["mse"], "psnr": r["psnr"], "ssim": r["ssim"],
        "sam": r["sam"], "sam_valid": r["sam_valid"],
        # P1
        "floor_psnr": p1["best_floor"]["psnr"], "floor_sam": p1["best_floor"]["sam"],
        "oracle_psnr": p1["identity_oracle"]["psnr"],
        "oracle_sam": p1["identity_oracle"]["sam"],
        "lift_psnr_db": p1["lift"]["psnr_db"], "lift_sam_rel": p1["lift"]["sam_relative"],
        "headroom_captured_psnr": p1["headroom_captured_psnr"],
        "P1": p1["verdict"],
        # P2
        "latent_elements": p2["latent_elements"], "compression_ratio": p2["compression_ratio"],
        "rate_dev_pct": p2["deviation_pct"], "rate_matched": p2["rate_matched"], "P2": p2["verdict"],
        # P3
        "active_units": p3["active_unit_fraction"], "latent_swap_delta": p3["latent_swap_delta"],
        "collapsed": p3["collapsed"], "P3": p3["verdict"],
        # P4-P7
        "sri": p4["sri"], "P4": p4["verdict"],
        "inpaint_gain": p5["relative_gain"], "P5": p5["verdict"],
        "mean_npr": p6["mean_npr"], "P6": p6["verdict"],
        "physics_r2": p7["physics_r2"], "scene_id_acc": p7["scene_id_accuracy"],
        "P7": p7["verdict"],
    }
    if "per_branch_mse" in p2:
        row.update({f"our_{k}": v for k, v in p2["per_branch_mse"].items()})
    return row


def pairwise(cells: list[dict], cfg: dict) -> list[dict]:
    """Every model pair within a dataset, on the shared patch sample."""
    st = cfg["p8_statistics"]
    rows = []
    by_ds: dict[str, list[dict]] = {}
    for c in cells:
        by_ds.setdefault(c["dataset"], []).append(c)

    for ds, cs in sorted(by_ds.items()):
        usable = [c for c in cs if c["verdict"] != "INVALID"]
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
                        metric, f"{a['model']}|{a['loss']}", f"{b['model']}|{b['loss']}",
                        va[:n], vb[:n], min_effect=min_eff, higher_is_better=hib,
                        cfg=st, seed=cfg["sampling"]["seed"]))
            for r in apply_holm(family, st):
                d = r.as_row(); d["dataset"] = ds
                rows.append(d)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    cols = list({k: None for r in rows for k in r})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def render_verdict(rows: list[dict], stats_rows: list[dict], cfg: dict) -> str:
    L = []
    A = L.append
    A("=" * 78)
    A(" FALSIFICATION SUITE — VERDICT")
    A(f" preregistered {cfg.get('registered_on')} · split {cfg.get('split')} "
      f"· {cfg['sampling']['max_patches']} patches/cell")
    A("=" * 78)
    A("")
    A("Probes: P1 trivial floors · P2 latent rate · P3 collapse · P4 spatial")
    A("        reliance · P5 spectral inpainting · P6 purification · P7 probe")
    A("Verdict INVALID = not a usable result (collapsed, or below a trivial")
    A("        predictor). Such cells are findings, not scores.")

    by_ds: dict[str, list[dict]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)

    for ds, rs in sorted(by_ds.items()):
        A("")
        A("=" * 78)
        A(f" {ds}")
        A("=" * 78)
        A("")
        A(f"  {'model|loss':<34}{'verdict':<9}{'PSNR':>7}{'SAMv':>8}"
          f"{'lift dB':>9}{'headrm':>8}  probes")
        A("  " + "-" * 74)
        for r in sorted(rs, key=lambda z: -z["psnr"]):
            probes = "".join(
                "." if r[p] == "PASS" else ("X" if r[p] in ("FAIL",) else "!")
                for p in ("P1", "P2", "P3", "P4", "P5", "P6", "P7"))
            A(f"  {r['model'] + '|' + r['loss']:<34}{r['verdict']:<9}"
              f"{r['psnr']:>7.2f}{r['sam_valid']:>8.4f}"
              f"{r['lift_psnr_db']:>9.2f}{r['headroom_captured_psnr']:>8.2f}  {probes}")
        A("  " + " " * 72 + "P1234567")

        inval = [r for r in rs if r["verdict"] == "INVALID"]
        if inval:
            A("")
            A("  NOT USABLE:")
            for r in inval:
                A(f"    {r['model']}|{r['loss']}: {r['reason']}")

        A("")
        A("  Latent rate (must be matched for the reconstruction comparison to mean anything):")
        for r in sorted(rs, key=lambda z: z["model"]):
            ok = "matched" if r["rate_matched"] else "UNMATCHED"
            A(f"    {r['model'] + '|' + r['loss']:<34}{r['latent_elements']:>9,} elements"
              f"  {r['compression_ratio']:>7.1f}:1  ({r['rate_dev_pct']:+5.1f}%)  {ok}")

        A("")
        A("  Mechanism probes:")
        A(f"    {'model|loss':<34}{'SRI':>8}{'inpaint':>9}{'NPR':>7}{'phys R2':>9}{'scene acc':>11}")
        for r in sorted(rs, key=lambda z: z["model"]):
            A(f"    {r['model'] + '|' + r['loss']:<34}{r['sri']:>8.3f}"
              f"{r['inpaint_gain']:>9.3f}{r['mean_npr']:>7.3f}"
              f"{r['physics_r2']:>9.3f}{r['scene_id_acc']:>11.3f}")
        A("      SRI     <0.02 => uses no spatial context (expected for vae-1d)")
        A("      inpaint <0.10 => no spectral prior; copies the input through")
        A("      NPR     <0.90 => genuinely denoises; ~1.0 => passes noise through")
        A("      phys R2 >=0.5 => latent encodes absorption chemistry")

        ours = [r for r in rs if r["model"] == "vae-our" and "our_mse_final" in r]
        if ours:
            o = ours[0]
            A("")
            A("  vae-our loss decomposition (total = final + 0.5*spatial + 0.5*spectral):")
            A(f"    mse_final    {o['our_mse_final']:.6f}   <- what it actually reconstructs")
            A(f"    mse_spatial  {o['our_mse_spatial']:.6f}   <- 256-dim whole-patch bottleneck")
            A(f"    mse_spectral {o['our_mse_spectral']:.6f}")
            A(f"    final is {100 * o['our_final_share_of_total']:.1f}% of the reported total.")
            if o["our_final_share_of_total"] < 0.5:
                A("    => the headline MSE is dominated by the AUXILIARY terms, not by")
                A("       reconstruction quality. Quote mse_final when comparing.")

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
    A(" A win counts only if the cell is VALID, the rate is matched, and the")
    A(" pairwise difference is both significant after Holm AND above the")
    A(" preregistered effect floor. 'significant_but_negligible' is not a win —")
    A(" with hundreds of paired patches almost any difference reaches p<0.05.")
    A(" Raw SAM is not comparable across datasets (near-zero pixels contribute")
    A(" exactly pi/2 whatever the model predicts); use SAMv.")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate probe results into CSV + verdict.")
    ap.add_argument("--probes-dir", type=Path, default=Path("results/probes"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--prereg", type=Path,
                    default=REPO_ROOT / "inference" / "preregistration.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.prereg.read_text())
    cells = load_cells(args.probes_dir)
    if not cells:
        print(f"no probe results under {args.probes_dir}")
        return 1

    rows = [flatten(c) for c in cells]
    stats_rows = pairwise(cells, cfg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "probes.csv")
    write_csv(stats_rows, args.out_dir / "stats.csv")
    txt = render_verdict(rows, stats_rows, cfg)
    (args.out_dir / "VERDICT.txt").write_text(txt + "\n")
    print(txt)
    print(f"\nwrote {args.out_dir}/probes.csv, stats.csv, VERDICT.txt "
          f"({len(rows)} cells, {len(stats_rows)} comparisons)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
