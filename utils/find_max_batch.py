"""
utils/find_max_batch.py
-----------------------
Find the largest batch size that fits a given VRAM budget, and time it.

One batch size PER DATASET
==========================
The ablation compares MODELS within a dataset, so that is the axis the batch
size must be held constant on: all 4 models x both loss regimes at a given
dataset train at the same batch size, and nothing about a row's result can be
attributed to it. Across datasets there is no controlled comparison to protect —
IIRS and M3 differ in band count, scene count, spatial sampling and SNR, and the
per-dataset YAMLs already vary `vae_3d_base_ch` and friends for param matching.

Holding one number across datasets as well would just waste GPU: activation
memory for the binding model scales roughly as `base_ch * C`, and M3 (84 bands)
is ~6x lighter than CRIMS (456 bands). A single global batch sized for CRIMS
leaves most of the card idle on M3.

So this reports ONE NUMBER PER DATASET = min over the models at that dataset
(--global restores a single number across everything).

The number must still hold across PLATFORMS for a given dataset, or within-
dataset fairness breaks the moment one slot runs on Kaggle and another on the
lab. So derive against the tightest budget any slot for that dataset will see:

  * lab    : 24 GB, single GPU                  <- binding
  * HPC    : 40 GB A100, single GPU             <- headroom
  * Kaggle : 2 x T4 15 GB with nn.DataParallel  <- per-device load is B/2,
             i.e. ~30 GB effective, so it clears if the lab's 24 GB does

Usage
=====
    # per-dataset numbers for the lab machine (what goes in the YAMLs)
    PYTHONPATH=. python utils/find_max_batch.py --budget-gb 24 --time

    # derive 24 GB numbers from a SMALLER card: measures peak at B=1,2,4 and
    # fits peak(B) = fixed + B * marginal, which is linear to within a percent.
    PYTHONPATH=. python utils/find_max_batch.py --budget-gb 24 --fit

    # one global number across all datasets (the old behaviour)
    PYTHONPATH=. python utils/find_max_batch.py --budget-gb 24 --global
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.registry import MODEL_NAMES, build_model  # noqa: E402
from utils.config import DATASETS, apply_dataset, settings  # noqa: E402
from utils.hyperparams import apply_hyperparams, load_hyperparams  # noqa: E402


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def try_batch(model_name: str, dataset: str, batch: int, device: torch.device,
              amp_dtype: torch.dtype, steps: int = 3) -> tuple[bool, float, float]:
    """
    Run `steps` real fwd+bwd+step iterations at this batch size.

    Returns (fits, peak_GB, seconds_per_step). A real optimizer step matters:
    AdamW's exp_avg/exp_avg_sq states are two extra copies of every parameter and
    are only allocated on the first step, so a forward-only probe underestimates
    the true peak.
    """
    _free()
    try:
        apply_dataset(dataset)
        apply_hyperparams(settings, load_hyperparams(dataset))
        model = build_model(model_name).to(device)
        # Materialize LazyLinear before measuring.
        with torch.no_grad():
            model(torch.randn(2, settings.input_height, settings.input_width,
                              settings.input_channels, device=device))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        # Always probe the physics path: it computes SAM on top of MSE+KLD, so
        # it is the heavier of the two regimes and gives the worst-case peak.
        use_physics = True

        x = torch.randn(batch, settings.input_height, settings.input_width,
                        settings.input_channels, device=device)
        _free()

        t_total = 0.0
        for i in range(steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"),
                                    dtype=amp_dtype):
                terms = model.loss_terms(x, beta=1e-3, lambda_physics=0.3,
                                         use_physics=use_physics)
            terms["loss"].backward()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            # Skip the first step: it pays for cuDNN autotune and the one-off
            # allocation of AdamW's optimizer state.
            if i > 0:
                t_total += time.time() - t0

        peak = (torch.cuda.max_memory_allocated() / 1e9
                if device.type == "cuda" else 0.0)
        per_step = t_total / max(steps - 1, 1)
        del model, opt, x
        _free()
        return True, peak, per_step

    except torch.cuda.OutOfMemoryError:
        _free()
        return False, float("inf"), 0.0
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        _free()
        return False, float("inf"), 0.0


def search(model_name: str, dataset: str, budget_gb: float, device: torch.device,
           amp_dtype: torch.dtype, headroom: float, max_batch: int,
           verbose: bool = True) -> tuple[int, float, float]:
    """
    Largest power-of-two batch fitting `budget_gb * (1 - headroom)`.

    Powers of two only: they are what people actually report in papers, they keep
    the DataParallel split even, and a finer search would be false precision
    against a budget that varies with driver version and fragmentation anyway.
    """
    limit = budget_gb * (1.0 - headroom)
    best, best_peak, best_step = 0, 0.0, 0.0
    b = 1
    while b <= max_batch:
        fits, peak, per_step = try_batch(model_name, dataset, b, device, amp_dtype)
        if verbose:
            mark = "ok " if (fits and peak <= limit) else "NO "
            shown = "OOM" if peak == float("inf") else f"{peak:6.2f} GB"
            print(f"      B={b:<4d} {mark} peak {shown}"
                  + (f"  {per_step * 1000:7.1f} ms/step" if fits else ""))
        if not fits or peak > limit:
            break
        best, best_peak, best_step = b, peak, per_step
        b *= 2
    return best, best_peak, best_step


def fit_max_batch(model_name: str, dataset: str, budget_gb: float,
                  device: torch.device, amp_dtype: torch.dtype, headroom: float,
                  max_batch: int, probes=(1, 2, 4),
                  verbose: bool = True) -> tuple[int, float, float]:
    """
    Estimate the max batch for `budget_gb` by measuring small batches and
    extrapolating, so a 24 GB answer can be derived on a smaller card.

    Peak memory is `fixed + B * marginal` to within about a percent: weights,
    gradients and AdamW state are batch-independent, while activations scale
    linearly with B. Least-squares fit over `probes`, then solve for the budget
    and round DOWN to a power of two.

    Returns (batch, predicted_peak_GB, seconds_per_sample).
    """
    limit = budget_gb * (1.0 - headroom)
    xs, ys, ts = [], [], []
    for b in probes:
        fits, peak, per_step = try_batch(model_name, dataset, b, device, amp_dtype)
        if not fits:
            break
        xs.append(b); ys.append(peak); ts.append(per_step / b)
    if not xs:
        return 0, 0.0, 0.0

    if len(xs) >= 2:
        marginal, fixed = np.polyfit(xs, ys, 1)
    else:
        marginal, fixed = ys[0], 0.0
    marginal = max(marginal, 1e-6)

    raw = (limit - fixed) / marginal
    b = 1
    while b * 2 <= min(raw, max_batch):
        b *= 2
    b = max(b, 0 if raw < 1 else 1)

    if verbose:
        pts = "  ".join(f"B={x}:{y:.2f}GB" for x, y in zip(xs, ys))
        print(f"      {pts}")
        print(f"      fit: {fixed:.2f} GB fixed + {marginal:.3f} GB/sample"
              f"  ->  {limit:.1f} GB usable fits B<={raw:.0f}")
    return b, fixed + b * marginal, (sum(ts) / len(ts))


def main() -> int:
    ap = argparse.ArgumentParser(description="Find the largest batch size fitting a VRAM budget.")
    ap.add_argument("--budget-gb", type=float, default=24.0,
                    help="Per-GPU VRAM budget in GB (lab 24, HPC 40, one T4 15).")
    ap.add_argument("--headroom", type=float, default=0.15,
                    help="Fraction of the budget left free for fragmentation/cuDNN workspace.")
    ap.add_argument("--model", action="append", choices=list(MODEL_NAMES),
                    help="Model to probe (repeatable). Default: all.")
    ap.add_argument("--dataset", action="append", choices=sorted(DATASETS),
                    help="Dataset to probe (repeatable). Default: all.")
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--fit", action="store_true",
                    help="Extrapolate from B=1,2,4 instead of doubling until OOM. "
                         "Lets a small GPU predict the answer for --budget-gb 24.")
    ap.add_argument("--global", dest="global_", action="store_true",
                    help="Report ONE number across all datasets instead of one "
                         "per dataset. Only correct if you intend to hold batch "
                         "size fixed across sensors too.")
    ap.add_argument("--time", action="store_true",
                    help="Also print ms/step and projected epoch time at the chosen batch.")
    ap.add_argument("--patches-per-epoch", type=int, default=None,
                    help="For the epoch-time projection (default: settings.train_patch_cap).")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device — this must run on the target GPU.")
        return 2

    device = torch.device("cuda")
    major, _ = torch.cuda.get_device_capability()
    amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    models = args.model or list(MODEL_NAMES)
    datasets = args.dataset or sorted(DATASETS)
    n_per_epoch = args.patches_per_epoch or settings.train_patch_cap or 7000

    print("=" * 72)
    print(f" GPU        : {torch.cuda.get_device_name(0)} ({total_gb:.1f} GB physical)")
    print(f" budget     : {args.budget_gb:.1f} GB, {args.headroom:.0%} headroom "
          f"-> {args.budget_gb * (1 - args.headroom):.1f} GB usable")
    print(f" amp dtype  : {str(amp_dtype).replace('torch.', '')}")
    if args.fit:
        print(" mode       : FIT (measure B=1,2,4, extrapolate) — safe on a small card")
    print(f" sizing     : {'ONE number for everything' if args.global_ else 'ONE number PER DATASET'}")
    print("=" * 72)

    results: dict[tuple[str, str], tuple[int, float, float]] = {}
    for ds in datasets:
        for m in models:
            print(f"\n  {m} | {ds}")
            probe = fit_max_batch if args.fit else search
            best, peak, per_unit = probe(m, ds, args.budget_gb, device, amp_dtype,
                                         args.headroom, args.max_batch)
            results[(m, ds)] = (best, peak, per_unit)
            if best == 0:
                print("    -> does NOT fit even at B=1")
                continue
            line = f"    -> max B={best} ({'predicted' if args.fit else 'peak'} {peak:.2f} GB)"
            if args.time and per_unit:
                # search() returns s/step at `best`; fit_max_batch returns s/sample.
                per_sample = per_unit / best if not args.fit else per_unit
                line += f", ~{n_per_epoch * per_sample / 60:.1f} min/epoch at {n_per_epoch:,} patches"
            print(line)

    if all(v[0] == 0 for v in results.values()):
        print("\n nothing fits — lower the model widths or the budget is wrong.")
        return 1

    print("\n" + "=" * 72)
    oom = [f"{m}|{d}" for (m, d), v in results.items() if v[0] == 0]

    if args.global_:
        fitting = {k: v for k, v in results.items() if v[0] > 0}
        binding = min(fitting, key=lambda k: fitting[k][0])
        b = fitting[binding][0]
        print(f" binding pair : {binding[0]} | {binding[1]}")
        print(f" GLOBAL B*    : {b}   (same value in all four YAMLs)")
    else:
        print(" PER-DATASET B* — paste into each hyperparam-config-<DS>.yaml:\n")
        print(f"   {'dataset':<8} {'B*':>5}  {'binding model':<24} {'pred GB':>8}")
        print(f"   {'-' * 50}")
        for ds in datasets:
            row = {m: results[(m, ds)] for m in models if results[(m, ds)][0] > 0}
            if not row:
                print(f"   {ds:<8} {'-':>5}  nothing fits")
                continue
            bind = min(row, key=lambda m: row[m][0])
            b, peak, _ = row[bind]
            print(f"   {ds:<8} {b:>5}  {bind:<24} {peak:>7.2f}")
        print("\n   Held constant across all 4 models x 2 loss regimes WITHIN each")
        print("   dataset — that is the axis the ablation compares on. It must also")
        print("   hold across platforms for a given dataset, so derive it against")
        print("   the tightest budget any slot for that dataset will run under.")
        print("   Kaggle 2xT4 + nn.DataParallel sees B/2 per device.")

    if oom:
        print(f"\n WARNING: does not fit at B=1: {', '.join(oom)}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
