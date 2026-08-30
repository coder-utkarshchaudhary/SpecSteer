"""
inference/downstream.py
-----------------------
Downstream latent-space experiments for the HSI VAE ablation study.

These probe whether a trained VAE's latent space is *ready for a diffusion model*
without actually training one, by exercising the latent manifold directly. Both
experiments are model-agnostic: they only use the registry contract's
``encode_latents(x) -> list[Tensor]`` / ``decode_latents(list[Tensor]) -> (B,H,W,C)``
(see modules/registry.py), so vae-our and all three baselines run through the
same code path.

Experiment 1 — Latent Noise-Injection Robustness ("does the manifold collapse?")
    Encode a clean batch, add Gaussian noise eps ~ N(0, sigma^2) directly to the
    latent tensors at several sigma levels, decode, and measure reconstruction
    quality (SAM / PSNR / SSIM) vs the clean input. A robust latent space
    degrades gracefully; a fragile one hallucinates catastrophically at moderate
    sigma. Hypothesis: baselines collapse by sigma=0.5; vae-our degrades slowly.

Experiment 2 — Chemical Interpolation Smoothness ("is the manifold smooth?")
    Take the latents of two endpoint pixels (e.g. an ice-like and a rock-like
    spectrum), linearly interpolate z_mix = alpha*z_A + (1-alpha)*z_B for
    alpha in [0,1], decode, and measure how smoothly the decoded spectrum
    changes. A generative-ready manifold yields a smooth spectral transition
    (small second-difference / low jaggedness); baselines jump erratically.

Run from the repo root with PYTHONPATH set:
    PYTHONPATH=. python inference/downstream.py --dataset IIRS
    PYTHONPATH=. python inference/downstream.py --dataset IIRS --models vae-our vae-standard
    python inference/downstream.py --help

Results are printed as comparison tables and (with --save-plots) written under
--out-dir as JSON + PNG figures.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
# Repo root must come FIRST: this file's own directory is sys.path[0] when run as
# a script, and it contains inference.py, which would otherwise shadow the
# `inference` package and break `from inference.inference import ...`.
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from inference.inference import compute_mse, compute_psnr, compute_ssim, load_model  # noqa: E402
from modules.losses import spectral_angle_mapper_loss  # noqa: E402
from modules.registry import MODEL_NAMES, PHYSICS_ONLY, checkpoint_name, resolve_checkpoint  # noqa: E402
from utils.config import DATASETS, apply_dataset, settings  # noqa: E402
from utils.hyperparams import apply_hyperparams, load_hyperparams  # noqa: E402
from utils.logging_setup import get_console_logger, get_run_logger, timestamp  # noqa: E402
from utils.training.dataloader import build_dataloader  # noqa: E402


# Default sigma sweep for the noise-injection test (absolute, per the spec).
DEFAULT_SIGMAS = (0.0, 0.1, 0.5, 1.0)
# Default alpha grid for the interpolation test.
DEFAULT_N_ALPHA = 11


class _MultiLogger:
    """
    Thin adapter that fans out ``.info(...)``/``.warning(...)``/``.error(...)``
    calls to a list of underlying ``logging.Logger`` objects.

    Used by run_for_model so a single per-model call site logs to both stdout
    (via the shared console logger) and that model's on-disk file.
    """

    def __init__(self, loggers):
        self._loggers = list(loggers)

    def info(self, msg, *args, **kwargs):
        for lg in self._loggers:
            lg.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        for lg in self._loggers:
            lg.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        for lg in self._loggers:
            lg.error(msg, *args, **kwargs)


# ---------------------------------------------------------------------------
# Latent helpers (work on the opaque list[Tensor] returned by encode_latents)
# ---------------------------------------------------------------------------

def add_latent_noise(latents, sigma, generator=None):
    """Return a new latent list with N(0, sigma^2) noise added to every tensor."""
    out = []
    for z in latents:
        if sigma > 0:
            noise = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
            out.append(z + sigma * noise)
        else:
            out.append(z.clone())
    return out


def lerp_latents(latents_a, latents_b, alpha):
    """Elementwise z_mix = alpha*z_A + (1-alpha)*z_B for each tensor in the list."""
    return [alpha * za + (1.0 - alpha) * zb for za, zb in zip(latents_a, latents_b)]


def slice_latents(latents, idx):
    """Keep a single batch element (index idx) from each latent tensor, dim 0 preserved."""
    return [z[idx:idx + 1] for z in latents]


# ---------------------------------------------------------------------------
# Experiment 1 — Latent noise-injection robustness
# ---------------------------------------------------------------------------

def noise_injection(model, x, sigmas, generator=None):
    """
    Encode x, inject latent noise at each sigma, decode, and score vs clean x.

    Args:
        model     : a registry model (encode_latents / decode_latents contract)
        x         : clean batch (B, H, W, C), values in [0, 1]
        sigmas    : iterable of noise std-devs (0.0 == no noise, sanity baseline)
        generator : optional torch.Generator for reproducible noise

    Returns:
        list of dicts: {sigma, mse, sam, psnr, ssim} (metrics vs the clean input)
    """
    latents = model.encode_latents(x)
    results = []
    for sigma in sigmas:
        noisy = add_latent_noise(latents, sigma, generator=generator)
        recon = model.decode_latents(noisy)
        results.append({
            "sigma": float(sigma),
            "mse": compute_mse(x, recon),
            "sam": spectral_angle_mapper_loss(x, recon).item(),
            "psnr": compute_psnr(x, recon),
            "ssim": compute_ssim(x, recon),
        })
    return results


# ---------------------------------------------------------------------------
# Experiment 2 — Chemical interpolation smoothness
# ---------------------------------------------------------------------------

def _decoded_pixel_spectrum(recon, pixel):
    """Extract the (C,) spectrum at (row, col) from the first sample of recon."""
    r, c = pixel
    return recon[0, r, c].detach().cpu().numpy()


def interpolation_smoothness(model, x, idx_a, idx_b, n_alpha, pixel):
    """
    Interpolate between two samples' latents and measure spectral smoothness.

    The two endpoints (idx_a, idx_b in the batch) stand in for two distinct
    materials (e.g. ice vs basalt). We walk alpha from 0->1, decode each mix,
    and read the reflectance spectrum at ``pixel``.

    Smoothness metric: the mean L2 norm of the *second difference* of the decoded
    spectra along the alpha axis. A smooth manifold changes at a near-constant
    rate (small second difference); an erratic one jumps (large second
    difference). We also report the total-variation path length for context.

    Returns:
        dict(alphas, spectra (n_alpha, C), jaggedness, path_length)
    """
    latents = model.encode_latents(x)
    la = slice_latents(latents, idx_a)
    lb = slice_latents(latents, idx_b)

    alphas = np.linspace(0.0, 1.0, n_alpha)
    spectra = []
    for alpha in alphas:
        mix = lerp_latents(la, lb, float(alpha))
        recon = model.decode_latents(mix)
        spectra.append(_decoded_pixel_spectrum(recon, pixel))
    spectra = np.stack(spectra, axis=0)          # (n_alpha, C)

    # Second difference along alpha: s[k-1] - 2 s[k] + s[k+1]; mean L2 over steps.
    if n_alpha >= 3:
        second_diff = spectra[:-2] - 2.0 * spectra[1:-1] + spectra[2:]
        jaggedness = float(np.mean(np.linalg.norm(second_diff, axis=-1)))
    else:
        jaggedness = float("nan")

    # Total variation path length: sum of consecutive step magnitudes.
    steps = np.linalg.norm(np.diff(spectra, axis=0), axis=-1)
    path_length = float(np.sum(steps))

    return {
        "alphas": alphas.tolist(),
        "spectra": spectra,
        "jaggedness": jaggedness,
        "path_length": path_length,
    }


# ---------------------------------------------------------------------------
# Per-model driver
# ---------------------------------------------------------------------------

def resolve_ckpt(model_name, dataset, loss, ckpt_dir, seed=None, select="sam"):
    """Locate one cell's checkpoint, honouring the seed axis and the two
    selection criteria (see modules/registry.py:resolve_checkpoint)."""
    return resolve_checkpoint(ckpt_dir, dataset, model_name, loss,
                              seed=seed, select=select)


def run_for_model(model_name, x, args, device, generator, logger):
    """Load one model's checkpoint and run both experiments; return a result dict."""
    # vae-our is physics-only; the baselines use the requested loss regime.
    loss = "physics" if model_name in PHYSICS_ONLY else args.loss
    ckpt_file = (Path(args.ckpt) if args.ckpt else
                 resolve_ckpt(model_name, args.dataset, loss, args.ckpt_dir,
                              seed=args.seed, select=args.select))
    if not ckpt_file.exists():
        logger.warning(f"[skip] {model_name}: checkpoint not found ({ckpt_file})")
        return None

    model, _ = load_model(model_name, ckpt_file, device)

    noise = noise_injection(model, x, args.sigmas, generator=generator)
    interp = interpolation_smoothness(
        model, x, args.idx_a, args.idx_b, args.n_alpha, tuple(args.pixel)
    )
    return {"model": model_name, "loss": loss, "ckpt": str(ckpt_file),
            "noise": noise, "interp": interp}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _broadcast(loggers, level, msg):
    """Emit ``msg`` at ``level`` to every logger in ``loggers``."""
    for lg in loggers:
        lg.log(level, msg)


def print_noise_table(all_results, sigmas, loggers):
    _broadcast(loggers, logging.INFO, "=" * 72)
    _broadcast(loggers, logging.INFO, " EXPERIMENT 1 — Latent noise-injection robustness (metrics vs clean x)")
    _broadcast(loggers, logging.INFO, "=" * 72)
    for metric in ("psnr", "ssim", "sam"):
        arrow = "higher=better" if metric in ("psnr", "ssim") else "lower=better, rad"
        _broadcast(loggers, logging.INFO, f" {metric.upper()}  ({arrow}) by sigma:")
        header = f"  {'model':24s}" + "".join(f"{f'σ={s:g}':>12s}" for s in sigmas)
        _broadcast(loggers, logging.INFO, header)
        for r in all_results:
            by_sigma = {d["sigma"]: d[metric] for d in r["noise"]}
            row = f"  {r['model']:24s}" + "".join(f"{by_sigma.get(float(s), float('nan')):12.4f}" for s in sigmas)
            _broadcast(loggers, logging.INFO, row)


def print_interp_table(all_results, loggers):
    _broadcast(loggers, logging.INFO, "=" * 72)
    _broadcast(loggers, logging.INFO, " EXPERIMENT 2 — Chemical interpolation smoothness")
    _broadcast(loggers, logging.INFO, "=" * 72)
    _broadcast(loggers, logging.INFO, f"  {'model':24s}{'jaggedness↓':>16s}{'path_length':>16s}")
    for r in all_results:
        it = r["interp"]
        _broadcast(
            loggers, logging.INFO,
            f"  {r['model']:24s}{it['jaggedness']:16.5f}{it['path_length']:16.5f}",
        )
    _broadcast(loggers, logging.INFO,
                "  jaggedness = mean L2 of 2nd difference of decoded spectra along alpha")
    _broadcast(loggers, logging.INFO,
                "  (lower = smoother chemical transition = more generative-ready manifold)")


def save_outputs(all_results, args, loggers):
    out_dir = Path(args.out_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON dump (spectra arrays converted to lists).
    serializable = []
    for r in all_results:
        rc = {k: v for k, v in r.items() if k != "interp"}
        it = dict(r["interp"])
        it["spectra"] = np.asarray(it["spectra"]).tolist()
        rc["interp"] = it
        serializable.append(rc)
    (out_dir / "downstream_results.json").write_text(json.dumps(serializable, indent=2))
    _broadcast(loggers, logging.INFO, f"Saved metrics JSON to {out_dir / 'downstream_results.json'}")

    if not args.save_plots:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _broadcast(loggers, logging.WARNING, "[warn] matplotlib not available; skipping plots.")
        return

    # Plot 1: PSNR-vs-sigma degradation curves (all models on one axis).
    plt.figure(figsize=(7, 5))
    for r in all_results:
        s = [d["sigma"] for d in r["noise"]]
        p = [d["psnr"] for d in r["noise"]]
        plt.plot(s, p, marker="o", label=r["model"])
    plt.xlabel("latent noise σ"); plt.ylabel("PSNR (dB)")
    plt.title(f"Latent noise-injection robustness — {args.dataset}")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "noise_robustness_psnr.png", dpi=150); plt.close()

    # Plot 2: interpolation spectra (one subplot per model) at the chosen pixel.
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, r in zip(axes[0], all_results):
        it = r["interp"]
        spectra = np.asarray(it["spectra"])
        for k, alpha in enumerate(it["alphas"]):
            ax.plot(spectra[k], color=plt.cm.viridis(alpha), alpha=0.8)
        ax.set_title(f"{r['model']}\njagged={it['jaggedness']:.4f}")
        ax.set_xlabel("band"); ax.set_ylabel("reflectance (norm.)")
    plt.suptitle(f"Chemical interpolation (α: 0→1) — {args.dataset}, pixel {tuple(args.pixel)}")
    plt.tight_layout(); plt.savefig(out_dir / "interpolation_spectra.png", dpi=150); plt.close()
    _broadcast(loggers, logging.INFO, f"Saved plots to {out_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latent-space downstream experiments (noise robustness + interpolation)."
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES),
                        help="Which trained models to compare (default: all four).")
    parser.add_argument("--loss", default="physics", choices=["standard", "physics"],
                        help="Loss regime of the baseline checkpoints to load "
                             "(vae-our is always physics).")
    parser.add_argument("--data-root", default=None, help="Override processed root.")
    parser.add_argument("--packed-root", default=None,
                        help="Override the packed-shard dir (data/packed/<DS>).")
    parser.add_argument("--ckpt-dir", default="model", help="Checkpoint root (per-dataset subfolders).")
    parser.add_argument("--seed", type=int, default=None,
                    help="Which training seed's CHECKPOINT to evaluate. Omit when only one seed exists; required once several do, since picking implicitly would make the result depend on file order. (RNG seed for the noise draws is --rng-seed.)")
    parser.add_argument("--select", choices=("sam", "mse"), default="sam",
                    help="Which checkpoint to load: the epoch selected on best val SAM (default) or on best val reconstruction MSE. Every cell writes both; a comparison must read the SAME criterion for every model.")
    parser.add_argument("--ckpt", default=None,
                        help="Explicit checkpoint path (single-model runs only).")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of clean patches to run the experiments on.")
    parser.add_argument("--num-workers", type=int, default=settings.num_workers)

    # Experiment 1 knobs
    parser.add_argument("--sigmas", type=float, nargs="+", default=list(DEFAULT_SIGMAS),
                        help="Latent noise std-devs to sweep (0.0 = clean sanity check).")

    # Experiment 2 knobs
    parser.add_argument("--idx-a", type=int, default=0, help="Endpoint A sample index in the batch.")
    parser.add_argument("--idx-b", type=int, default=1, help="Endpoint B sample index in the batch.")
    parser.add_argument("--n-alpha", type=int, default=DEFAULT_N_ALPHA,
                        help="Number of interpolation steps (alpha grid size).")
    parser.add_argument("--pixel", type=int, nargs=2, default=[32, 32],
                        help="(row col) pixel whose spectrum is tracked during interpolation.")

    parser.add_argument("--rng-seed", type=int, default=42,
                        help="Seed for the latent-noise / interpolation RNG (not the checkpoint seed).")
    parser.add_argument("--out-dir", default="results/downstream")
    parser.add_argument("--save-plots", action="store_true", help="Write PNG figures under --out-dir.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ckpt is not None and len(args.models) != 1:
        raise SystemExit("--ckpt only makes sense with a single --models entry.")

    # verify=True cross-checks the configured band count against what is
    # actually on disk, so a mismatch fails here with both numbers rather
    # than as an opaque assert inside SpectralBranch.forward.
    apply_dataset(args.dataset, verify=True, processed_root=args.data_root)
    # Per-dataset latent-rate / capacity knobs the models were trained with;
    # without this the models rebuild at the dataclass defaults and
    # load_state_dict fails. Mirrors train/train.py and inference.py.
    apply_hyperparams(settings, load_hyperparams(args.dataset))
    torch.manual_seed(args.rng_seed)
    generator = torch.Generator(device=device).manual_seed(args.rng_seed)

    ts = timestamp()
    console = get_console_logger()
    # Build one file-only logger per model so each log is a self-contained
    # trace of that run. Shared messages (banner, comparison tables) are
    # broadcast to every file logger *and* printed once via ``console``.
    model_loggers = {}
    for model_name in args.models:
        loss_for_name = "physics" if model_name in PHYSICS_ONLY else args.loss
        model_loggers[model_name] = get_run_logger(
            "downstream", model_name, args.dataset,
            loss=loss_for_name, ts=ts, stream=False,
        )
    shared_loggers = [console] + list(model_loggers.values())

    _broadcast(shared_loggers, logging.INFO, "==============================================")
    _broadcast(shared_loggers, logging.INFO, f"  dataset  : {args.dataset} (C={settings.input_channels})")
    _broadcast(shared_loggers, logging.INFO, f"  models   : {args.models}")
    _broadcast(shared_loggers, logging.INFO, f"  split    : {args.split}  |  batch: {args.batch_size}")
    _broadcast(shared_loggers, logging.INFO, f"  sigmas   : {args.sigmas}")
    _broadcast(shared_loggers, logging.INFO, f"  device   : {device}")
    _broadcast(shared_loggers, logging.INFO, "==============================================")

    # One fixed clean batch shared across all models, so the comparison is fair.
    loader = build_dataloader(
        args.dataset, args.split,
        processed_root=args.data_root,
        packed_root=args.packed_root,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    x = next(iter(loader)).to(device)
    if x.shape[0] < 2:
        _broadcast(shared_loggers, logging.ERROR,
                   "Need at least 2 samples in the batch for interpolation endpoints.")
        raise SystemExit("Need at least 2 samples in the batch for interpolation endpoints.")
    args.idx_a = min(args.idx_a, x.shape[0] - 1)
    args.idx_b = min(args.idx_b, x.shape[0] - 1)

    all_results = []
    for model_name in args.models:
        # Fresh generator per model so each sees the same noise draw sequence.
        gen = torch.Generator(device=device).manual_seed(args.rng_seed)
        # For per-model progress ("[skip] ..."), pair the model's file logger
        # with the console so it also lands on stdout.
        per_model_loggers = [console, model_loggers[model_name]]
        adapter = _MultiLogger(per_model_loggers)
        r = run_for_model(model_name, x, args, device, gen, adapter)
        if r is not None:
            all_results.append(r)

    if not all_results:
        _broadcast(shared_loggers, logging.ERROR,
                   "No models could be evaluated (no checkpoints found).")
        raise SystemExit("No models could be evaluated (no checkpoints found).")

    print_noise_table(all_results, args.sigmas, shared_loggers)
    print_interp_table(all_results, shared_loggers)
    save_outputs(all_results, args, shared_loggers)


if __name__ == "__main__":
    main()
