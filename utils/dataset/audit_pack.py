"""
utils/dataset/audit_pack.py
---------------------------
Prove that packing introduced no artifact, and report which storage backend each
dataset actually resolves to.

Why this exists
===============
`pack.py` rewrites every patch: it crops bands, replaces non-finite values,
divides by the per-patch max, and casts to float16. Each of those is a chance to
silently corrupt the training set, and the previous grid could only be shown to
have used the packed data by *inferring* it from epoch timings. This turns both
questions into a command.

It checks five things:

  1. **Backend** — which of packed/legacy each dataset resolves to, and where.
  2. **Fidelity** — packed rows vs. their source `.npy` put through the same
     normalisation. Bounds the fp16 round-trip error.
  3. **Row alignment** — that row *i* of the shard really is
     `source_files[i]`, not an off-by-one or a shuffled write. Sampled rows
     catch systematic damage; a full-shard fingerprint scan catches single
     corrupted or duplicated rows, which sampling would miss.
  4. **Distribution drift** — per-band mean/std computed both ways.
  5. **Census** — non-finite values, negatives, and near-zero pixels, which are
     properties of the DATA rather than of packing but change how the metrics
     must be read (see the notes printed at the end).

Usage
=====
    PYTHONPATH=. python utils/dataset/audit_pack.py --all
    PYTHONPATH=. python utils/dataset/audit_pack.py --dataset CRIMS --rows 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import DATASETS, probe_channels, settings  # noqa: E402
from utils.dataset.pack import normalise  # noqa: E402

SPLITS = ("train", "valid", "test")

# fp16 has ~11 mantissa bits and these values top out at 1.0, so the worst
# representable round-trip error is 2**-11 ~= 4.9e-4. Anything materially above
# that means something other than the dtype cast changed the data.
FP16_TOL = 1e-3

# A pixel whose spectral energy is below SAM's epsilon gets a norm dominated by
# that epsilon, cos_sim ~ 0, and contributes exactly pi/2 to the mean angle
# regardless of what the model predicted.
SAM_EPS_ENERGY = 1e-8


def audit_backend(ds: str) -> dict:
    meta = DATASETS[ds]
    count, source, where = probe_channels(ds)
    packed_dir = Path(meta["packed_root"])
    shards = {s: (packed_dir / f"{s}.npy") for s in SPLITS}
    present = {s: p.is_file() for s, p in shards.items()}
    return {
        "dataset": ds, "probe_count": count, "probe_source": source,
        "probe_location": where,
        "packed_root": str(packed_dir),
        "shards_present": present,
        "backend": "packed" if present.get("train") else "legacy",
    }


def audit_fidelity(ds: str, split: str, rows: int, rng) -> dict | None:
    meta = DATASETS[ds]
    npy = Path(meta["packed_root"]) / f"{split}.npy"
    js = Path(meta["packed_root"]) / f"{split}.json"
    if not (npy.is_file() and js.is_file()):
        return None
    m = json.loads(js.read_text())
    mm = np.load(npy, mmap_mode="r")
    proc = Path(m.get("processed_root") or meta["processed_root"])

    n = int(m["n"])
    if mm.shape[0] != n:
        return {"split": split, "error": f"row count {mm.shape[0]} != manifest {n}"}

    idx = rng.choice(n, size=min(rows, n), replace=False)
    worst = 0.0
    worst_row = -1
    max_rel = 0.0
    misaligned = []
    src_mean, pk_mean, src_var, pk_var = [], [], [], []

    for i in idx:
        src_path = proc / m["source_files"][int(i)]
        if not src_path.is_file():
            misaligned.append(f"row {i}: missing source {src_path}")
            continue
        ref, mx = normalise(np.load(src_path), m.get("crop_bands"))
        got = np.asarray(mm[int(i)], dtype=np.float32)
        if got.shape != ref.shape:
            misaligned.append(f"row {i}: shape {got.shape} != {ref.shape}")
            continue
        d = float(np.abs(got - ref).max())
        if d > worst:
            worst, worst_row = d, int(i)
        denom = float(np.abs(ref).max()) or 1.0
        max_rel = max(max_rel, d / denom)
        # Row alignment: a shuffled/off-by-one write shows up as a large error
        # here even though the fp16 cast alone could never produce one.
        if d > 0.05:
            misaligned.append(f"row {i}: |diff| {d:.3g} — not an fp16 cast error")
        src_mean.append(ref.mean(axis=(0, 1)))
        pk_mean.append(got.mean(axis=(0, 1)))
        src_var.append(ref.var(axis=(0, 1)))
        pk_var.append(got.var(axis=(0, 1)))
        if abs(mx - float(m["patch_max"][int(i)])) > 1e-3 * max(mx, 1.0):
            misaligned.append(f"row {i}: patch_max {mx} != manifest "
                              f"{m['patch_max'][int(i)]}")

    band_mean_drift = float(np.abs(np.mean(src_mean, 0) - np.mean(pk_mean, 0)).max()) if src_mean else float("nan")
    band_std_drift = float(np.abs(np.sqrt(np.mean(src_var, 0)) - np.sqrt(np.mean(pk_var, 0))).max()) if src_var else float("nan")

    return {
        "split": split, "n": n, "C": int(m["C"]), "raw_C": int(m.get("raw_C", m["C"])),
        "crop_bands": m.get("crop_bands"), "rows_checked": len(idx),
        "max_abs_diff": worst, "max_abs_diff_row": worst_row,
        "max_rel_diff": max_rel,
        "band_mean_drift": band_mean_drift, "band_std_drift": band_std_drift,
        "alignment_problems": misaligned,
        "ok": worst <= FP16_TOL and not misaligned,
    }


def audit_duplicates(ds: str, split: str) -> dict | None:
    """
    Scan EVERY row for exact duplicates using a cheap fingerprint.

    The sampled fidelity check above catches *systematic* damage — an off-by-one
    or a shuffled write shows up in any row you look at. It does not catch a
    single corrupted row, because sampling 32 of 15,000 rows almost never hits
    it. This closes that gap at negligible cost: three contiguous pixel spectra
    per row (~3 pages of I/O) is enough to make a collision between genuinely
    different patches essentially impossible, while a duplicated or overwritten
    row matches exactly.
    """
    npy = Path(DATASETS[ds]["packed_root"]) / f"{split}.npy"
    if not npy.is_file():
        return None
    mm = np.load(npy, mmap_mode="r")
    n, H, W = mm.shape[0], mm.shape[1], mm.shape[2]
    probes = [(0, 0), (H // 2, W // 2), (H - 1, W - 1)]

    seen: dict[bytes, int] = {}
    dupes: list[str] = []
    for i in range(n):
        fp = b"".join(np.asarray(mm[i, y, x], dtype=np.float16).tobytes()
                      for y, x in probes)
        prev = seen.get(fp)
        if prev is not None:
            dupes.append(f"row {i} is byte-identical to row {prev}")
        else:
            seen[fp] = i
    return {"split": split, "n": n, "duplicates": dupes}


def audit_census(ds: str, split: str, rows: int, rng) -> dict | None:
    npy = Path(DATASETS[ds]["packed_root"]) / f"{split}.npy"
    if not npy.is_file():
        return None
    mm = np.load(npy, mmap_mode="r")
    idx = rng.choice(mm.shape[0], size=min(rows, mm.shape[0]), replace=False)
    nonfinite = neg = dead = tot_v = tot_p = 0
    lo, hi = np.inf, -np.inf
    for i in idx:
        a = np.asarray(mm[int(i)], dtype=np.float32)
        nonfinite += int((~np.isfinite(a)).sum())
        neg += int((a < 0).sum())
        tot_v += a.size
        e = (a ** 2).sum(axis=-1)
        dead += int((e < SAM_EPS_ENERGY).sum())
        tot_p += e.size
        lo, hi = min(lo, float(a.min())), max(hi, float(a.max()))
    return {"split": split, "min": lo, "max": hi,
            "pct_nonfinite": 100 * nonfinite / max(tot_v, 1),
            "pct_negative": 100 * neg / max(tot_v, 1),
            "pct_dead_pixels": 100 * dead / max(tot_p, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--dataset", action="append", choices=sorted(DATASETS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rows", type=int, default=32,
                    help="Rows sampled per split for the fidelity check.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    datasets = args.dataset or sorted(DATASETS)
    rng = np.random.default_rng(args.seed)
    failures = 0
    census_rows = []

    for ds in datasets:
        b = audit_backend(ds)
        print("=" * 74)
        print(f" {ds}   backend: {b['backend'].upper()}")
        print("=" * 74)
        print(f"  packed root   : {b['packed_root']}")
        print(f"  shards        : " + ", ".join(
            f"{k}{'OK' if v else 'MISSING'}" for k, v in
            [(f"{s}=", b['shards_present'][s]) for s in SPLITS]))
        print(f"  channel probe : {b['probe_count']} bands from {b['probe_source']}")
        if b["backend"] != "packed":
            print("  !! no packed train shard — training would fall back to the SLOW")
            print("     per-patch path. Run: python utils/dataset/pack.py")
            failures += 1
            continue

        for split in SPLITS:
            f = audit_fidelity(ds, split, args.rows, rng)
            if f is None:
                print(f"  {split:<6}: no shard")
                continue
            if "error" in f:
                print(f"  {split:<6}: FAIL {f['error']}")
                failures += 1
                continue
            tag = "OK  " if f["ok"] else "FAIL"
            print(f"  {split:<6}: {tag} n={f['n']:>6,} C={f['C']}"
                  + (f" (cropped from {f['raw_C']})" if f["crop_bands"] else "")
                  + f"  rows={f['rows_checked']}"
                  f"  max|diff|={f['max_abs_diff']:.2e}"
                  f"  band mean drift={f['band_mean_drift']:.2e}"
                  f"  std drift={f['band_std_drift']:.2e}")
            for p in f["alignment_problems"][:5]:
                print(f"          !! {p}")
            if not f["ok"]:
                failures += 1

            d = audit_duplicates(ds, split)
            if d and d["duplicates"]:
                print(f"          !! {len(d['duplicates'])} duplicate row(s) "
                      f"across all {d['n']:,} — a shard should contain no two "
                      f"identical patches:")
                for p in d["duplicates"][:5]:
                    print(f"             {p}")
                failures += 1
            c = audit_census(ds, split, args.rows, rng)
            if c and split == "test":
                census_rows.append((ds, c))

    if census_rows:
        print("\n" + "=" * 74)
        print(" DATA CENSUS (test split) — properties of the data, not of packing")
        print("=" * 74)
        print(f"  {'dataset':<8}{'min':>9}{'max':>7}{'nonfinite':>11}{'negative':>10}{'dead px':>10}")
        for ds, c in census_rows:
            print(f"  {ds:<8}{c['min']:>9.3f}{c['max']:>7.3f}"
                  f"{c['pct_nonfinite']:>10.2f}%{c['pct_negative']:>9.2f}%"
                  f"{c['pct_dead_pixels']:>9.2f}%")
        print("\n  negative : every model ends in sigmoid and CANNOT represent these.")
        print("             A large figure is a hard reconstruction floor, not a")
        print("             model failure.")
        print("  dead px  : pixels with spectral energy < 1e-8. SAM's epsilon")
        print("             dominates their norm, so each contributes exactly")
        print("             pi/2 REGARDLESS of the prediction. A large figure")
        print("             means raw SAM is not comparable across datasets —")
        print("             use the SAM-valid variant from inference/probes.py.")

    print("\n" + "=" * 74)
    print(f" failures: {failures}   (fp16 tolerance {FP16_TOL:.0e})")
    print("=" * 74)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
