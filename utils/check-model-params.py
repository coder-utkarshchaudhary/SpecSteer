"""
utils/check-model-params.py
---------------------------
Fair-ablation parameter audit.

For every (model, dataset) pair, applies the dataset settings + per-dataset
hyperparam YAML overrides, builds the model, runs a dummy forward to
materialize LazyLinear layers, and prints total parameter count. Emits a
4x3 table where every row should read within ~3% of vae-our per column.

Also solves for the baseline capacity knobs: `--solve` binary-searches each
baseline's width so its parameter count lands within tolerance of vae-our on
that dataset, and prints YAML ready to paste into
utils/hyperparam_configs/hyperparam-config-<DS>.yaml.

Re-run `--solve` after ANY change to modules/vae_*.py or to the spectral/spatial
width knobs in utils/config.py — the widths currently in the YAMLs were solved
against a different vae-our and will be stale.

Usage (from repo root, with venv active):
    python utils/check-model-params.py            # audit the current YAMLs
    python utils/check-model-params.py --solve    # re-derive the widths
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.registry import MODELS, MODEL_NAMES, build_model  # noqa: E402
from utils.config import DATASETS, apply_dataset, settings  # noqa: E402
from utils.hyperparams import apply_hyperparams, load_hyperparams  # noqa: E402


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_and_count(model_name: str, dataset: str, **overrides) -> int:
    """
    Parameter count for one (model, dataset), with optional Settings overrides.

    Only vae-our has LazyLinear layers, so only it needs the dummy forward to
    materialize — and that forward is the slow part (a 3D baseline forward on CPU
    takes tens of seconds). Skipping it for the baselines makes --solve's binary
    search fast enough to be practical.
    """
    apply_dataset(dataset)
    hp = load_hyperparams(dataset)
    apply_hyperparams(settings, hp)
    for k, v in overrides.items():
        setattr(settings, k, v)

    model = build_model(model_name)
    if any(isinstance(m, torch.nn.modules.lazy.LazyModuleMixin)
           for m in model.modules()):
        dummy = torch.randn(2, settings.input_height, settings.input_width,
                            settings.input_channels)
        with torch.no_grad():
            model(dummy)
    return count_params(model)


# Which Settings knob scales each baseline's capacity, and how to build a
# candidate value from a single scalar width.
_WIDTH_KNOB = {
    "vae-standard": ("vae_standard_base_ch", lambda w: w),
    "vae-3d-spatio-spectral": ("vae_3d_base_ch", lambda w: w),
    # vae-1d's three hidden dims stay in the 4:2:1 ratio the original picks used.
    "vae-1d-pixelwise": ("vae_1d_hidden_dims", lambda w: (w * 4, w * 2, w)),
}


def solve_width(model_name: str, dataset: str, target: int,
                lo: int = 4, hard_cap: int = 4096) -> tuple[int, int]:
    """
    Smallest-error width for `model_name` at `dataset`.

    Every baseline's parameter count is monotonically increasing in its width
    knob, so this is exponential search for a bracket followed by bisection.

    The ramp matters: starting bisection at a fixed high bound would *build* a
    model at that width, and e.g. vae-3d at base_ch=4096 is a 2-billion-parameter
    Conv3d stack that gets the process OOM-killed before it can measure anything.
    Ramping from `lo` only ever builds models near the answer.

    Returns (width, resulting_param_count).
    """
    knob, make = _WIDTH_KNOB[model_name]
    best: tuple[int, int] | None = None

    def measure(w: int) -> int:
        nonlocal best
        n = build_and_count(model_name, dataset, **{knob: make(w)})
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (w, n)
        return n

    # Ramp up until we overshoot the target (or hit the safety cap).
    hi = lo
    while measure(hi) < target:
        if hi >= hard_cap:
            return best
        lo, hi = hi, min(hi * 2, hard_cap)

    # Bracket is (lo, hi]; bisect it.
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if measure(mid) < target:
            lo = mid
        else:
            hi = mid
    return best


def audit() -> None:
    dataset_names = sorted(DATASETS)
    print(f"{'model':<24} " + "  ".join(f"{d:>14}" for d in dataset_names) + "   vs vae-our")
    print("-" * (24 + 16 * len(dataset_names) + 20))

    # No `if os.path.exists(...)` guard here: build_and_count feeds the model a
    # torch.randn dummy and never touches disk, so a dataset that isn't staged
    # on this machine still audits fine. (The guard used to index a str as a
    # dict -> TypeError, and skipping a dataset would leave reference[ds] unset
    # -> KeyError in the ratio loop below.)
    reference: dict[str, int] = {ds: build_and_count("vae-our", ds)
                                 for ds in dataset_names}

    for name in MODEL_NAMES:
        counts = {ds: build_and_count(name, ds) for ds in dataset_names}
        cells = "  ".join(f"{counts[ds]:>14,}" for ds in dataset_names)
        ratios = "  ".join(
            f"{ds}:{100 * (counts[ds] - reference[ds]) / reference[ds]:+.1f}%"
            for ds in dataset_names
        )
        print(f"{name:<24} {cells}   {ratios}")


def solve(tolerance: float) -> None:
    """Re-derive every baseline width and print paste-ready YAML."""
    dataset_names = sorted(DATASETS)
    for ds in dataset_names:
        target = build_and_count("vae-our", ds)
        print(f"\n# --- {ds}: vae-our target = {target:,} ({target / 1e6:.2f}M) ---")
        print(f"# paste into utils/hyperparam_configs/hyperparam-config-{ds}.yaml")
        worst = 0.0
        for name in _WIDTH_KNOB:
            knob, make = _WIDTH_KNOB[name]
            w, n = solve_width(name, ds, target)
            err = 100 * (n - target) / target
            worst = max(worst, abs(err))
            value = make(w)
            rendered = (f"[{', '.join(str(v) for v in value)}]"
                        if isinstance(value, tuple) else str(value))
            print(f"{knob}: {rendered}  # {n:,} ({err:+.2f}%)")
        flag = "" if worst <= tolerance else f"   <-- WORST {worst:.2f}% EXCEEDS {tolerance}%"
        print(f"# worst deviation: {worst:.2f}%{flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fair-ablation parameter audit.")
    ap.add_argument("--solve", action="store_true",
                    help="Re-derive baseline widths to match vae-our; print YAML.")
    ap.add_argument("--tolerance", type=float, default=3.0,
                    help="Percent deviation considered acceptable (default 3).")
    args = ap.parse_args()
    if args.solve:
        solve(args.tolerance)
    else:
        audit()


if __name__ == "__main__":
    main()
