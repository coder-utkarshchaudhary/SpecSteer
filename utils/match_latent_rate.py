"""
utils/match_latent_rate.py
--------------------------
Solve each model's bottleneck knob so every model in the ablation encodes to the
same LATENT BUDGET, and verify that the config actually achieves it.

Why this exists
===============
The ablation controls for two independent resources:

    parameters  -> capacity of the function class   (utils/check-model-params.py)
    latent rate -> width of the information bottleneck   (this file)

Only the first was being controlled. Rate was left to float by **512x** between
models, and for an autoencoder that is the more important of the two: for a
fixed architecture, reconstruction quality is bounded by rate almost by
definition, so a 2:1 bottleneck beats a 1024:1 one regardless of what is inside
it. Worse, on M3 `vae-our`'s latent was 1.52x LARGER than the cube it encodes —
an over-complete code that can copy the input outright, and unusable as an LDM
backbone (Stable Diffusion's AutoencoderKL is 48:1).

What is NOT equalised
=====================
Latent *shape*. `vae-standard` is a spatial grid with no spectral axis;
`vae-our` is a global vector beside a full-resolution per-pixel spectral map.
That geometry IS the architecture under test — forcing a common shape would
destroy the thing being measured. Only the scalar COUNT is matched.

Ordering
========
Run this BEFORE `check-model-params.py --solve`. Changing the latent changes the
parameter count (vae-standard's 1x1 projections grow with `latent_ch`;
vae-our's LazyLinear shrinks with `spectral_latent_dim`), so the parameter
solution must be recomputed against the new latents, not the other way round.

Usage
=====
    # solve for 64:1 and print paste-ready YAML
    PYTHONPATH=. python utils/match_latent_rate.py --ratio 64

    # verify the CURRENT config is matched to within tolerance
    PYTHONPATH=. python utils/match_latent_rate.py --ratio 64 --check

    # show what the current config actually encodes to, no target
    PYTHONPATH=. python utils/match_latent_rate.py --report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.registry import MODEL_NAMES, build_model  # noqa: E402
from utils.config import DATASETS, apply_dataset, settings  # noqa: E402
from utils.hyperparams import apply_hyperparams, load_hyperparams  # noqa: E402

# The knob that sets each model's bottleneck width, and the closed form for the
# latent element count as a function of it. The closed forms are used for the
# solve (fast); measure_latent() then confirms them against the real model, so a
# drift between this file and modules/ is caught rather than silently shipped.
_RATE_KNOB = {
    "vae-standard": "vae_standard_latent_ch",
    "vae-3d-spatio-spectral": "vae_3d_latent_ch",
    "vae-1d-pixelwise": "vae_1d_latent_dim",
    "vae-our": "spectral_latent_dim",
}


def latent_elements_closed_form(model_name: str, s) -> int:
    """Predicted latent scalar count per patch, from the config alone."""
    H, W, C = s.input_height, s.input_width, s.input_channels
    if model_name == "vae-standard":
        g = H // (2 ** s.vae_standard_n_down)
        return s.vae_standard_latent_ch * g * g
    if model_name == "vae-3d-spatio-spectral":
        mult = 2 ** s.vae_3d_n_down
        c_pad = -(-C // mult) * mult
        g = H // mult
        return s.vae_3d_latent_ch * (c_pad // mult) * g * g
    if model_name == "vae-1d-pixelwise":
        return H * W * s.vae_1d_latent_dim
    if model_name == "vae-our":
        # spatial global vector + full-resolution per-pixel spectral map
        return s.latent_dim + s.spectral_latent_dim * H * W
    raise ValueError(model_name)


# The quantisation step of each model's latent, in elements. A model can only
# land on integer multiples of its own grain, and the grains are mutually
# indivisible, so no single target is reachable by all four.
def latent_grain(model_name: str, s) -> int:
    """Elements added per unit increment of this model's rate knob."""
    base = latent_elements_closed_form(model_name, _with(s, _RATE_KNOB[model_name], 0))
    one = latent_elements_closed_form(model_name, _with(s, _RATE_KNOB[model_name], 1))
    return one - base


def common_target(s, quantum: int = None) -> int:
    """
    The common latent budget T for this dataset.

    Chosen as the nearest multiple of H*W (4096) to the 64:1 point. That is not
    arbitrary: 4096 is `vae-1d`'s grain (one latent channel per pixel) and it is
    divisible by `vae-standard`'s grain of 64, so BOTH of those models land on T
    exactly. `vae-3d`'s grain is (C_pad/8)*64, whose odd factor (11, 53, 57)
    makes an exact common multiple astronomical, and `vae-our` carries a
    `latent_dim`-sized global vector on top of a multiple of 4096 -- so those two
    land near T rather than on it, and their deviation is reported per cell.

    Solving for a fixed 64:1 RATIO instead is what left M3's vae-1d 23.8% off:
    5,376 is not a multiple of 4,096, so the model nearest the target could only
    reach 4,096. Solving for a reachable T inverts that.
    """
    quantum = quantum or (s.input_height * s.input_width)
    ideal = s.input_height * s.input_width * s.input_channels / 64.0
    return max(quantum, round(ideal / quantum) * quantum)


def effective_elements(model_name: str, elements: int, s) -> int:
    """
    Latent elements that actually encode real bands.

    Only `vae-3d` differs from its nominal count: it zero/replicate-pads the
    spectral depth up to a multiple of 2**n_down before encoding, so a fraction
    C/C_pad of its latent depth corresponds to padding rather than data. At the
    current padding that fraction is 100% everywhere except M3 (95.5%). It would
    drop to 65.6% on M3 if the depth were padded to a power of two, which is why
    that route to an exact match was rejected: it matches the nominal number
    while mismatching the thing the number measures.
    """
    if model_name != "vae-3d-spatio-spectral":
        return elements
    mult = 2 ** s.vae_3d_n_down
    c_pad = -(-s.input_channels // mult) * mult
    return int(round(elements * s.input_channels / c_pad))


@torch.no_grad()
def measure_latent(model_name: str, dataset: str, **overrides) -> int:
    """Ground truth: build the model and count elements in encode_latents()."""
    apply_dataset(dataset)
    apply_hyperparams(settings, load_hyperparams(dataset))
    for k, v in overrides.items():
        setattr(settings, k, v)
    settings.__post_init__()

    model = build_model(model_name)
    x = torch.randn(1, settings.input_height, settings.input_width,
                    settings.input_channels)
    if any(isinstance(m, torch.nn.modules.lazy.LazyModuleMixin)
           for m in model.modules()):
        model(x)
    return int(sum(t.numel() for t in model.encode_latents(x)))


def solve_knob(model_name: str, dataset: str, target: int) -> tuple[int, int]:
    """
    Smallest-error knob value for `target` latent elements.

    Every model's latent count is linear in its knob, so this divides rather than
    searching. Returns (knob_value, achieved_elements).
    """
    apply_dataset(dataset)
    apply_hyperparams(settings, load_hyperparams(dataset))
    knob = _RATE_KNOB[model_name]

    # Elements at knob=1, minus any knob-independent term (vae-our's 256-dim
    # spatial vector), gives the per-unit slope.
    base = latent_elements_closed_form(model_name, _with(settings, knob, 0))
    per_unit = latent_elements_closed_form(model_name, _with(settings, knob, 1)) - base
    value = max(1, round((target - base) / per_unit))
    achieved = latent_elements_closed_form(model_name, _with(settings, knob, value))
    return value, achieved


def _with(s, knob: str, value: int):
    setattr(s, knob, value)
    s.__post_init__()
    return s


def _reset(dataset: str) -> dict:
    """
    Restore `settings` to the dataset's on-disk config and return the pristine
    rate knobs.

    `apply_hyperparams` only writes keys that are actually present in the YAML,
    and the rate knobs are not there yet — so a knob mutated by a previous
    solve would survive the reset and leak into the next dataset's "before"
    reading. Snapshotting them explicitly is what keeps each row independent.
    """
    apply_dataset(dataset)
    apply_hyperparams(settings, load_hyperparams(dataset))
    return {k: getattr(settings, k) for k in set(_RATE_KNOB.values())}


def _restore(pristine: dict) -> None:
    for k, v in pristine.items():
        setattr(settings, k, v)
    settings.__post_init__()


def main() -> int:
    ap = argparse.ArgumentParser(description="Match every model to a common latent budget.")
    ap.add_argument("--ratio", type=float, default=64.0,
                    help="Target compression, input:latent (default 64, close to "
                         "Stable Diffusion's AutoencoderKL at 48:1). Ignored with --exact.")
    ap.add_argument("--exact", action="store_true",
                    help="Solve for a common reachable BUDGET T (nearest multiple of "
                         "H*W to the 64:1 point) instead of a common ratio. Lands "
                         "vae-standard and vae-1d exactly on T; the other two report "
                         "their deviation. This is the mode the grid uses.")
    ap.add_argument("--tolerance", type=float, default=None,
                    help="Percent deviation still counted as matched "
                         "(default: 10 with --exact, 25 otherwise).")
    ap.add_argument("--check", action="store_true",
                    help="Verify the CURRENT config is matched; exit non-zero if not.")
    ap.add_argument("--report", action="store_true",
                    help="Just show what the current config encodes to.")
    ap.add_argument("--verify-against-models", action="store_true", default=True,
                    help="Cross-check the closed forms by building each model.")
    args = ap.parse_args()
    tolerance = args.tolerance if args.tolerance is not None else (10.0 if args.exact else 25.0)

    datasets = sorted(DATASETS)
    models = list(MODEL_NAMES)
    failures = 0

    def target_for(s) -> int:
        return common_target(s) if args.exact else int(
            s.input_height * s.input_width * s.input_channels / args.ratio)

    mode = ("common BUDGET (nearest multiple of H*W to 64:1)" if args.exact
            else f"common RATIO {args.ratio:g}:1")

    # ---- current state -----------------------------------------------------
    print("=" * 92)
    print(f" CURRENT latent budget per patch   [{mode}, tolerance +/-{tolerance:g}%]")
    print("=" * 92)
    hdr = f"{'dataset':<8}{'input':>10}{'T':>9}{'ratio':>8} | " + "".join(
        f"{m.split('-')[1][:6]:>15}" for m in models)
    print(hdr + "\n" + "-" * len(hdr))

    for ds in datasets:
        pristine = _reset(ds)
        inp = settings.input_height * settings.input_width * settings.input_channels
        target = target_for(settings)
        cells = []
        for m in models:
            _restore(pristine)
            got = latent_elements_closed_form(m, settings)
            if args.verify_against_models:
                real = measure_latent(m, ds, **pristine)
                if real != got:
                    print(f"\n  !! closed form disagrees with the model: {m}|{ds} "
                          f"predicted {got:,}, measured {real:,}")
                    print("     utils/match_latent_rate.py is out of sync with modules/ — fix it.")
                    failures += 1
                    got = real
                _reset(ds)
                _restore(pristine)
            dev = 100 * (got - target) / target
            flag = "" if abs(dev) <= tolerance else "!"
            cells.append(f"{got:>9,}{dev:>+5.1f}%{flag}")
            if abs(dev) > tolerance:
                failures += 1
        print(f"{ds:<8}{inp:>10,}{target:>9,}{inp / target:>7.1f}:1 | "
              + "".join(f"{c:>15}" for c in cells))

    if args.report:
        return 0
    if args.check:
        print()
        if failures:
            print(f"NOT MATCHED: {failures} cell(s) outside +/-{tolerance:g}%. "
                  f"Run without --check for the knob values.")
        else:
            print(f"MATCHED: every model within +/-{tolerance:g}% of the common budget.")
        return 1 if failures else 0

    # ---- solve -------------------------------------------------------------
    print("\n" + "=" * 92)
    print(" SOLVED knobs — paste into utils/hyperparam_configs/hyperparam-config-<DS>.yaml")
    print("=" * 92)
    worst_overall = 0.0
    for ds in datasets:
        pristine = _reset(ds)
        inp = settings.input_height * settings.input_width * settings.input_channels
        target = target_for(settings)
        print(f"\n# --- {ds}: input {inp:,}, common budget T = {target:,} "
              f"({inp / target:.1f}:1) ---")
        worst = 0.0
        for m in models:
            knob = _RATE_KNOB[m]
            _restore(pristine)
            before_knob = getattr(settings, knob)
            value, achieved = solve_knob(m, ds, target)
            _restore(pristine)
            dev = 100 * (achieved - target) / target
            worst = max(worst, abs(dev))
            eff = effective_elements(m, achieved, settings)
            eff_note = "" if eff == achieved else f", {eff:,} effective after depth padding"
            exact = "EXACT" if achieved == target else f"{dev:+.1f}%"
            note = "unchanged" if value == before_knob else f"was {before_knob}"
            print(f"{knob}: {value}".ljust(34)
                  + f"# {achieved:,} elements ({exact}){eff_note} — {note}")
        worst_overall = max(worst_overall, worst)
        status = "OK" if worst <= tolerance else f"EXCEEDS +/-{tolerance:g}%"
        print(f"# worst deviation: {worst:.1f}%  {status}")

    print(f"\n# worst deviation across all datasets: {worst_overall:.1f}%")
    print("\n" + "=" * 92)
    print(" NEXT, in this order — both matter:")
    print("   1. paste the knobs above into the four YAMLs")
    print("   2. python utils/check-model-params.py --solve   (params AFTER rate)")
    print("   3. python utils/find_max_batch.py --budget-gb 20 --fit  (batch AFTER both)")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
