"""
inference/inference_missing_pixels.py
-------------------------------------
Missing-pixel recovery for the paper's Table 6. A random 10% of each patch's
pixels are zeroed across ALL bands (a transmission dropout), the corrupted
patch is reconstructed, and the output is scored against the CLEAN patch —
next to the same model's reconstruction of the uncorrupted patch.

Deliberately small, unlike inference_final.py: one checkpoint seed (67), one
mask ratio (0.10), one mask draw (RNG 67 — the same generator call as the
first draw of the earlier 3-draw run, so the masks are identical to it).

Scope: IIRS/AVIRIS/CRIMS x the same 21 cells as inference_final.py
(vae-our-nl physics-only; the three baselines x {physics, standard}),
select=sam, full combined valid+test split.

One CSV, results/final/missing_pixels/missing-pixel-recovery.csv:
    psnr_clean / psnr_masked  whole-cube PSNR from the POOLED MSE (the
                              reconstruction-quality.csv / Table 3 convention)
    psnr_drop                 psnr_clean - psnr_masked
    sam_valid_*               energy-masked SAM, radians (pixels whose clean
                              spectrum is below the SAM epsilon are excluded)

Run from the repo root:
    PYTHONPATH=. python inference/inference_missing_pixels.py --dry-run
    PYTHONPATH=. python inference/inference_missing_pixels.py
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
# Repo root first: this directory contains inference.py, which would otherwise
# shadow the `inference` package (same guard as inference_final.py).
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from modules.metrics import compute_mse  # noqa: E402
from modules.registry import resolve_checkpoint  # noqa: E402
from utils.config import apply_dataset, settings  # noqa: E402
from utils.hyperparams import apply_cli_overrides, apply_hyperparams, load_hyperparams  # noqa: E402

from inference.inference import load_model, sam_valid_sums, compute_psnr_from_mse  # noqa: E402
from inference.probes import load_prereg  # noqa: E402
from inference.inference_final import (  # noqa: E402
    DATASETS_SCOPE, MODELS_SCOPE, Logger, audit_dataset, build_inference_set,
    resolve_packed_root, cell_grid, _make_loader, write_csv,
)

CKPT_SEED = 67
MASK_RATIO = 0.10
MASK_SEED = 67
CACHE_VERSION = 1

CSV_NAME = "missing-pixel-recovery.csv"
CSV_COLUMNS = [
    "dataset", "model", "loss", "ckpt_seed", "mask_ratio",
    "psnr_clean", "psnr_masked", "psnr_drop",
    "sam_valid_clean", "sam_valid_masked", "sam_valid_drop",
    "n_samples",
]


class _Scores:
    """Pooled MSE (-> PSNR once, never averaged in dB) + pixel-pooled SAM-valid."""

    def __init__(self, min_energy: float):
        self.min_energy = min_energy
        self.mse_wsum = 0.0
        self.sam_valid_sum = 0.0
        self.n_valid_px = 0
        self.n = 0

    def update(self, x: torch.Tensor, recon: torch.Tensor) -> None:
        b = x.shape[0]
        self.mse_wsum += compute_mse(x, recon) * b
        sv_sum, sv_n, _ = sam_valid_sums(x, recon, self.min_energy)
        self.sam_valid_sum += sv_sum
        self.n_valid_px += sv_n
        self.n += b

    def psnr(self) -> float:
        return float(compute_psnr_from_mse(self.mse_wsum / max(self.n, 1)))

    def sam_valid(self) -> float:
        return self.sam_valid_sum / self.n_valid_px if self.n_valid_px else float("nan")


def evaluate_cell(model, loader, mask_np: np.ndarray, device, min_energy: float,
                  desc: str) -> dict:
    clean, masked = _Scores(min_energy), _Scores(min_energy)
    i = 0
    with torch.inference_mode():
        for x in tqdm(loader, desc=desc, leave=False):
            x = x.to(device, non_blocking=True)
            b = x.shape[0]
            m = torch.from_numpy(mask_np[i:i + b]).to(device)              # (b, H, W) bool
            x_masked = torch.where(m.unsqueeze(-1), torch.zeros_like(x), x)
            clean.update(x, model.reconstruct(x))
            masked.update(x, model.reconstruct(x_masked))                   # scored vs CLEAN x
            del x, m, x_masked
            i += b
    psnr_c, psnr_m = clean.psnr(), masked.psnr()
    sam_c, sam_m = clean.sam_valid(), masked.sam_valid()
    return {
        "psnr_clean": psnr_c, "psnr_masked": psnr_m, "psnr_drop": psnr_c - psnr_m,
        "sam_valid_clean": sam_c, "sam_valid_masked": sam_m, "sam_valid_drop": sam_m - sam_c,
        "n_samples": clean.n,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", default="model")
    p.add_argument("--out-dir", default="results/final/missing_pixels")
    p.add_argument("--packed-root", default=None,
                   help="COMMON packed root, one level above the per-dataset dirs "
                        "(e.g. data/packed). Default: DATASETS[ds]['packed_root'].")
    p.add_argument("--datasets", default=",".join(DATASETS_SCOPE))
    p.add_argument("--models", default=",".join(MODELS_SCOPE))
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute every cell even if a cached result exists.")
    p.add_argument("--no-telegram", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the cell grid + dataset audit; load no model.")
    p.add_argument("--set", action="append", default=None, metavar="KEY=VALUE")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    datasets = [d.strip().upper() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / CSV_NAME

    log = Logger(enabled=not args.no_telegram)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_prereg()
    min_energy = cfg["p1_trivial_floors"]["sam_valid_min_energy"]

    cells = cell_grid(models, datasets)
    log.send(f"Missing-pixel recovery started — {len(cells)} cells, ckpt seed {CKPT_SEED}, "
             f"mask ratio {MASK_RATIO}, mask seed {MASK_SEED}, device={device}, out={csv_path}")

    if args.dry_run:
        for ds, model, loss in cells:
            print(f"  {ds:8}{model:24}{loss}")
        for ds in datasets:
            try:
                audit_dataset(ds, resolve_packed_root(ds, args.packed_root), log)
            except FileNotFoundError as e:
                print(f"  WARNING: {e}")
        print(f"{CSV_NAME}: {','.join(CSV_COLUMNS)}")
        return 0

    rows: list[dict] = []
    t0 = time.time()
    had_failure = False
    for ds in datasets:
        try:
            packed_root = resolve_packed_root(ds, args.packed_root)
            audit_dataset(ds, packed_root, log)
            apply_dataset(ds, verify=True)
            apply_hyperparams(settings, load_hyperparams(ds))
            overrides = apply_cli_overrides(settings, args.set)
            if overrides:
                log.send(f"--set overrides active for {ds}: {overrides}")
            inference_set, _ = build_inference_set(ds, packed_root)
            batch_size = args.batch_size or min(settings.batch_size, 16)
            num_workers = args.num_workers if args.num_workers is not None else settings.num_workers
            H, W = settings.input_height, settings.input_width
            mask_np = np.random.default_rng(MASK_SEED).random((len(inference_set), H, W)) < MASK_RATIO
        except Exception as e:  # noqa: BLE001
            had_failure = True
            log.send_pre(f"❌ Missing-pixel - dataset setup FAILED on {ds}: {html.escape(str(e))}",
                         traceback.format_exc()[-2000:])
            continue

        for model_name, loss in [(m, l) for d, m, l in cells if d == ds]:
            tag = f"{ds} | {model_name} | {loss}"
            cache_path = cache_dir / f"{ds}__{model_name}__{loss}.json"
            if not args.overwrite and cache_path.is_file():
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    if cached.get("version") == CACHE_VERSION:
                        rows.append(cached["row"])
                        write_csv(rows, CSV_COLUMNS, csv_path)
                        log.send(f"Missing-pixel - {tag} - LOADED FROM CACHE, skipping")
                        continue
                except Exception as e:  # noqa: BLE001
                    print(f"Failed to read cache {cache_path}: {e}")

            log.send(f"Missing-pixel - START - {tag}")
            try:
                ckpt_path = resolve_checkpoint(args.ckpt_dir, ds, model_name, loss,
                                               seed=CKPT_SEED, select="sam")
                if not ckpt_path.is_file():
                    raise FileNotFoundError(f"missing checkpoint {ckpt_path}")
                model, _ = load_model(model_name, ckpt_path, device)
                loader = _make_loader(inference_set, batch_size, num_workers)
                res = evaluate_cell(model, loader, mask_np, device, min_energy, desc=tag)
                del model
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                had_failure = True
                log.send_pre(f"❌ Missing-pixel - {tag} FAILED: {html.escape(str(e))}",
                             traceback.format_exc()[-2000:])
                continue

            row = {"dataset": ds, "model": model_name, "loss": loss,
                   "ckpt_seed": CKPT_SEED, "mask_ratio": MASK_RATIO, **res}
            rows.append(row)
            cache_path.write_text(json.dumps({"version": CACHE_VERSION, "row": row}, indent=2),
                                  encoding="utf-8")
            write_csv(rows, CSV_COLUMNS, csv_path)
            log.send(f"Missing-pixel - DONE - {tag}: PSNR clean {res['psnr_clean']:.2f} / "
                     f"masked {res['psnr_masked']:.2f} / drop {res['psnr_drop']:.2f} dB; "
                     f"SAM-valid clean {res['sam_valid_clean']:.4f} / masked "
                     f"{res['sam_valid_masked']:.4f} / drop {res['sam_valid_drop']:.4f} rad")

    if rows:
        write_csv(rows, CSV_COLUMNS, csv_path)
        log.send_document(csv_path, caption=CSV_NAME)
        log.send_csv_as_text(csv_path, "Missing-pixel recovery (10% pixels, seed 67; SAMv rad)", [
            ("model", "model"), ("loss", "loss"),
            ("psnr_clean", "PSNR_clean"), ("psnr_masked", "PSNR_masked"), ("psnr_drop", "PSNR_drop"),
            ("sam_valid_clean", "SAMv_clean"), ("sam_valid_masked", "SAMv_masked"),
            ("sam_valid_drop", "SAMv_drop"),
        ])
    log.send(f"Missing-pixel recovery complete — {len(rows)}/{len(cells)} cells, "
             f"{(time.time() - t0) / 60:.1f} min")
    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
