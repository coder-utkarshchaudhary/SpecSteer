# Pipeline Status — Dual-Stream Physics-Informed VAE for HSI

> **Last updated:** 2026-08-21  
> **Status:** All 4 ablation models + downstream experiments implemented.
> IITD HPC (PBS Pro) launcher + reverse-tunnel Telegram relay added.
> **Wall-time overhaul landed** — see [§10 Performance](#10-performance--why-the-grid-was-slow).
> The 5-epoch smoke run took 1h14m–4h58m per slot; the same work now projects to
> well under the 5–6 h/run budget at 30 epochs.  
> **Latent rate matched to 64:1** across all four models — see
> [§11 Controls](#11-experimental-controls--rate-and-parameters). The previous
> config left rate floating 512x between models, and on M3 `vae-our`'s latent
> was *larger than its input*.
> **Falsification suite added** — see [§12 Probes](#12-falsification-suite).
> ⚠️ Compile-pass ≠ verified-run. See [Caveats](#7-caveats) before running.

---

## 1. Architecture Overview

```
data/original/<folder>/
  *_rfl_d18_srd.qub   ← raw reflectance cube (BSQ, float32, 256×H×250)
  *_rfl_d18_srd.hdr   ← ENVI header (bands/lines/samples/interleave)

        ↓  utils/dataset/preprocess.py
        ↓  select bands 7:115, normalise by 1500 nm, Savitzky-Golay smooth

        ↓  utils/dataset/slice.py
        ↓  region-disjoint 70/15/15 split → 64×64 patches at stride 48

data/processed/<folder>/
  train/  patch_00000.npy … (64, 64, 108) float32
  valid/  patch_00000.npy …
  test/   patch_00000.npy …

        ↓  utils/training/dataloader.py   → torch.utils.data.DataLoader

        ↓  train/train.py
               ┌─────────────────────────────────────────┐
               │      HSI_DualStream_PI_VAE              │
               │                                         │
               │  Spatial Branch          Spectral Branch│
               │  ─────────────          ───────────────│
               │  Conv1D (spectral→dim)   Conv1D × 2    │
               │  Conv2D × 4 (spatial↓)  1→32→64 ch     │
               │  flatten → z_s          flatten → z_p  │
               │                                         │
               │  reparameterize (shared)                │
               │   chunk(2, dim=1) → mu, logvar → z     │
               │                                         │
               │  Decoder spatial ↑  Decoder spectral ↑ │
               │        recon_s          recon_p         │
               │                 ↓                       │
               │         Late Fusion (Linear)            │
               │              recon_final                │
               └─────────────────────────────────────────┘
               Loss = MSE(final+0.5*s+0.5*p) + β·KLD + λ·SAM

        ↓  wandb  (metrics + checkpoint)
```

---

## 2. File Map

| File | Role | Status |
|------|------|--------|
| `utils/config.py` | All hyper-parameters + derived dims | ✅ Rewritten |
| `utils/dataset/preprocess.py` | load → select bands → normalise → smooth | ✅ New |
| `utils/dataset/slice.py` | preprocess → region-split → patch → save | ✅ New |
| `utils/dataset/pack.py` | patches → **capped fp16 memmap shards** (`data/packed/`) | ✅ New |
| `utils/dataset/inspect_channels.py` | verify on-disk band counts vs config | ✅ New |
| `utils/training/dataloader.py` | Packed + legacy backends, DataLoader factory | ✅ Rewritten |
| `utils/find_max_batch.py` | per-dataset batch size for a VRAM budget (`--fit` extrapolates) | ✅ |
| `utils/match_latent_rate.py` | solve each model's bottleneck to a common latent rate | ✅ New |
| `utils/dataset/audit_pack.py` | prove packing introduced no artifact; report backend | ✅ New |
| `utils/check-model-params.py` | param audit **+ `--solve` for baseline widths** | ✅ Fixed |
| `modules/SpatialBranch.py` | Spatial encoder-decoder | ✅ Fixed |
| `modules/SpectralBranch.py` | Spectral encoder-decoder | ✅ Fixed |
| `modules/vae_our.py` | vae-our dual-stream PI-VAE (+ encode/decode_latents) | ✅ |
| `modules/vae_standard.py` | Baseline A: 2D spatial VAE (AutoencoderKL-style) | ✅ New |
| `modules/vae_3d.py` | Baseline B: 3D spatio-spectral VAE | ✅ New |
| `modules/vae_1d.py` | Baseline C: 1D pixelwise VAE | ✅ New |
| `modules/losses.py` | Shared SAM + KL loss primitives | ✅ |
| `modules/registry.py` | CLI name → model class; model contract | ✅ |
| `train/train.py` | Training loop, wandb, CLI, checkpointing | ✅ Rewritten |
| `inference/inference.py` | Reconstruction eval (MSE/SAM/PSNR/SSIM) | ✅ |
| `inference/downstream.py` | Latent noise-injection + interpolation experiments | ✅ |
| `inference/probes.py` | **Falsification suite — 7 probes on frozen models** | ✅ New |
| `inference/preregistration.yaml` | Thresholds, fixed before any run | ✅ New |
| `inference/stats.py` | Paired bootstrap / permutation / Holm / effect sizes | ✅ New |
| `inference/verdict.py` | probes.csv, stats.csv, VERDICT.txt | ✅ New |
| `docs/preregistration.md` | Why each threshold is what it is | ✅ New |
| `notebooks/*.ipynb` | Per-model self-contained training notebooks | ✅ |
| `inference/notebooks/*.ipynb` | Per-model self-contained eval notebooks | ✅ |
| `scripts/preprocess.sh` | One-command preprocessing runner | ✅ New |
| `scripts/train.sh` | Training runner (single run or full 28-run grid, `--overwrite`) | ✅ |
| `scripts/train_fixed.sh` | **pack → zip → smoke → full grid, one command** | ✅ New |
| `docs/file_processing.py` | Reference script (do not modify) | — |

### Ablation models & the model contract

All four models implement one model-agnostic contract (see `modules/registry.py`;
`train.py`/`inference.py` never branch on model type):

```
forward(x)                                               # x: (B, H, W, C)
loss_terms(x, beta, lambda_physics, use_physics) -> dict(loss, mse, kld, sam)
reconstruct(x) -> (B, H, W, C)
encode_latents(x) -> list[Tensor]                        # deterministic (mu) latents
decode_latents(list[Tensor]) -> (B, H, W, C)
```

| Model | Registry name | Latent (IIRS) | Hypothesis |
|-------|---------------|--------|------------|
| A: 2D Spatial | `vae-standard` | `(B, 256, 8, 8)` map | 2D convs blur pixel spectra → good PSNR, poor SAM |
| B: 3D Spatio-Spectral | `vae-3d-spatio-spectral` | `(B, 8, C/8, 8, 8)` volume | averages bands+pixels, param-heavy, collapse-prone |
| C: 1D Pixelwise | `vae-1d-pixelwise` | `(B, H, W, 4)` per-pixel | great chemistry (SAM), no spatial denoise → poor PSNR/SSIM |
| Proposed: PRISM | `vae-our` | `[(B,256), (B,4,H,W)]` | spatial+spectral isolation → high PSNR *and* low SAM |

Latent **shapes** differ by design — that geometry *is* what the ablation tests.
Latent **budgets** are matched to 64:1 (§11), and parameter counts to within
1.9 % (below). Both controls hold simultaneously.

Baselines A/B preserve the spatial grid at 8×8 (H,W ÷ 8) and reconstruct
exactly for any band count C. B now strides the spectral depth as well and pads
it up to a multiple of `2**n_down` before the encoder, cropping back after the
decoder (only M3 actually pads, 84 → 88); A/C are C-agnostic by construction.
All three run unchanged on IIRS/M3/AVIRIS/CRIMS.

All four models are matched to `vae-our`'s parameter count **per dataset**
(within 1.7%), via the width knobs in the hyperparam YAMLs. Regenerate those
widths after ANY architecture change with:

```bash
PYTHONPATH=. python utils/check-model-params.py --solve   # prints paste-ready YAML
PYTHONPATH=. python utils/check-model-params.py           # audit the result
```

| Dataset | vae-our target | `vae_standard_base_ch` | `vae_3d_base_ch` | `vae_1d_hidden_dims` |
|---|---|---|---|---|
| IIRS (C=256)   | 10.87 M | 86 | 45 | (2748, 1374, 687) |
| M3 (C=84)      | 10.70 M | 88 | 45 | (2856, 1428, 714) |
| AVIRIS (C=424) | 11.21 M | 85 | 46 | (2668, 1334, 667) |
| CRIMS (C=456)  | 11.28 M | 85 | 46 | (2656, 1328, 664) |

**Order matters when regenerating: rate first, then parameters.** Shrinking
`vae-our`'s spectral latent shrinks its `LazyLinear` (4096→256 becomes 4096→8),
so the parameter *target* moves. Solving parameters first and rate second
produces a stale match.

### Downstream experiments (`inference/downstream.py`)

Model-agnostic latent-space probes proving diffusion-readiness without training an LDM:

```bash
# All 4 models on IIRS test split, with figures:
PYTHONPATH=. python inference/downstream.py --dataset IIRS --save-plots
```

1. **Noise-injection robustness** — encode → add `N(0, σ²)` to latents at
   `σ ∈ {0, 0.1, 0.5, 1.0}` → decode → SAM/PSNR/SSIM vs clean. Robust manifolds
   degrade gracefully; fragile ones collapse by σ=0.5.
2. **Chemical interpolation smoothness** — `z_mix = α·z_A + (1−α)·z_B` for
   `α ∈ [0,1]`, decode, track a pixel's spectrum. Reports *jaggedness* (mean L2
   of the 2nd difference along α); lower = smoother = more generative-ready.

---

## 3. Configuration (`utils/config.py`)

All settings live in the `Settings` dataclass. Defaults are set for 64×64 patches
with 108-band IIRS cubes. Change values in the dataclass; derived fields
(`conv_output_*`, `spectral_*`) are recomputed automatically in `__post_init__`.

### Key config values

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `input_height/width` | 64 | patch spatial size |
| `input_channels` | 256 / 84 / 424 / 456 | per dataset (IIRS / M3 / AVIRIS / CRIMS) |
| `norm_target_nm` | 1500.0 | reference band resolved per-cube from its wavelengths |
| `savgol_window` | 7 | Savitzky-Golay window length |
| `patch_size / patch_stride` | 64 / 48 | 25% overlap |
| `split_ratios` | (0.70, 0.15, 0.15) | train/valid/test |
| **`train_patch_cap`** | **7000** | max training patches per dataset (applied by `pack.py`) |
| **`patch_cap_seed`** | **1234** | seed for the cap's scene-stratified subsample |
| `batch_size` | 32 / 32 / 16 / 16 | **per dataset** (IIRS / M3 / AVIRIS / CRIMS) — see §5 |
| `vae_standard_latent_ch` | 256 / 84 / 424 / 456 | latent-rate knob, per dataset |
| `vae_3d_latent_ch` | 8 | already exactly 64:1 — unchanged by the rate match |
| `vae_1d_latent_dim` | 4 / 1 / 7 / 7 | latent-rate knob, per dataset |
| `num_workers` | 8 | dataloader workers |
| `reduced_dims` | 32 | spatial Conv1D output channels |
| `latent_dim` | 256 | spatial latent (post-reparameterize) |
| `n_2D_conv_blocks` | 4 | spatial bottleneck: 64→4 px |
| `spectral_n_1D_conv_blocks` | 2 | spectral bottleneck |
| `spectral_latent_dim` | 4 / 1 / 7 / 7 | per-pixel spectral latent — **per dataset**, see §11 |
| **`spectral_base_ch`** | **32** | spectral Conv1D width — **decoupled from C**, see §10 |
| `spectral_transpose_c/l` | 64 / C÷4 | decoder reshape target |
| `spectral_linear_expansion_dim` | 16·C | 64 × (C÷4) |

**Band counts are verified against the data, not trusted.** `apply_dataset(verify=True)`
(used by `train.py`, `inference.py`, `downstream.py`) probes a real patch and
raises with both numbers on a mismatch. Run the standalone check any time:

```bash
PYTHONPATH=. python utils/dataset/inspect_channels.py
```

### Spectral dimension arithmetic

With `spectral_conv1D_kernel_size=4`, `stride=2`, `padding=1`:
```
Conv1d:          L_out = L_in // 2
ConvTranspose1d: L_out = 2 * L_in

IIRS (C=256, spectral_base_ch=32):
Encoder:  256 → 128 → 64  (L);  channels  1 → 32 → 64
Decoder:   64 → 128 → 256 (L);  channels 64 → 32 → 1
```

Sequence length `L` still tracks the band count, so the latent stays
sensor-aware. Only the **channel width** is now a free hyper-parameter
(`spectral_base_ch`) instead of being pinned to `input_channels`. That one
change is worth 35–97× in FLOPs — see §10.

---

## 4. Data Pipeline

### Step 1 — Preprocess & Slice

```bash
# Full pipeline (all 10 folders):
bash scripts/preprocess.sh

# With overwrite (redo existing output):
bash scripts/preprocess.sh --overwrite

# Custom paths:
DATA_ROOT=data/original OUT_ROOT=data/processed bash scripts/preprocess.sh
```

This will:
1. For each folder in `data/original/`: load the `*_rfl_d18_srd.qub` file.
2. Select bands `[7:115]` (108 bands).
3. Normalise each pixel spectrum by the ≈1500 nm reference band.
4. Smooth with Savitzky-Golay (window=7, poly=2) along the spectral axis.
5. Carve the cube height-wise into 70/15/15 contiguous regions.
6. Extract 64×64 patches at stride 48; drop partial edge patches.
7. Save each patch as `data/processed/<folder>/<split>/patch_NNNNN.npy`.

**Expected patch counts** (for H≈14k cubes, W=250):
- Per folder (train): ~1,015 patches  
- Per folder (valid/test): ~215 patches each  
- Total (10 folders): ~15,000 patches

### Step 2 — Pack (REQUIRED before training)

```bash
# all datasets, honouring settings.train_patch_cap (7000)
PYTHONPATH=. python utils/dataset/pack.py --verify

# one dataset, from scratch
PYTHONPATH=. python utils/dataset/pack.py --dataset CRIMS --overwrite --verify

# uncapped
PYTHONPATH=. python utils/dataset/pack.py --cap 0
```

Produces:

```
data/packed/<DATASET>/
  train.npy    (N, 64, 64, C) float16, per-patch max-normalised, band-cropped
  train.json   metadata + provenance (source file list, per-patch maxima)
  valid.npy / valid.json
  test.npy  / test.json
```

Four things happen here, and each fixes a distinct problem (§10):

1. **float16** halves the bytes. Values top out at 1.0 and fp16's ~5e-4 relative
   precision is *finer* than the bfloat16 autocast training already uses.
   Measured worst-case round-trip error: 2.4e-4.
2. **One file replaces ~15,000.** Per-file open overhead disappears and the OS
   page cache starts working across epochs.
3. **Normalisation moves here**, which permanently removes the full-dataset
   rescan the dataloader used to do at the start of *every* run.
4. **The band crop is applied here.** `crop_bands` is otherwise only honoured by
   `slice.py`, which never runs for CRIMS (it ships pre-processed) — so CRIMS's
   457 → 456 crop happens at pack time or not at all.

The training split is capped at `train_patch_cap` (7000), subsampled
proportionally across scenes with a fixed seed so every model sees the identical
subset. `valid`/`test` are never capped.

| | M3 | IIRS | AVIRIS | CRIMS |
|---|---|---|---|---|
| available | 19,746 | 14,624 | 11,027 | 2,561 |
| **trained on** | **7,000** | **7,000** | **7,000** | **2,561** |

Besides the 1.6–2.8× speedup, this equalises the training budget across sensors —
an improvement on the previous 8× imbalance, not a shortcut.

> **Where `data/packed/` lives is load-bearing.** fp16 + single-file alone buys
> ~2–3×. The projections in §10 assume it sits on **local NVMe** (or is held in
> RAM via `--cache-ram`), not back on the external drive the raw patches live on.
> Capped, all four train splits total ~53 GB; RAM caching peaks at ~25 GB since
> only one dataset is loaded at a time.

### Step 3 — Zip for Kaggle

`scripts/train_fixed.sh` does this automatically, or:

```bash
zip -0 -r -q dataset.zip data/packed     # -0 = store; fp16 sensor noise does not deflate
```

---

## 5. Training

### Prerequisites

```bash
# Install dependencies (already in .venv):
pip install torch torchvision scipy wandb

# One-time W&B login (run once; stores credentials in ~/.netrc):
wandb login
```

### Quick start — the whole thing, one command

```bash
bash scripts/train_fixed.sh
```

Runs: verify band counts → pack the capped fp16 dataset → zip it to
`dataset.zip` for Kaggle → 2-epoch smoke across all 28 slots → 30-epoch full
grid. Both training passes use `--overwrite`, so a leftover checkpoint can never
silently consume a slot. The smoke pass writes to `model_smoke/` so it cannot be
mistaken for a real result, and the full grid is **not** launched if the smoke
pass reports any failure.

```bash
bash scripts/train_fixed.sh --dry-run          # print the plan, run nothing
bash scripts/train_fixed.sh --pack-only        # build + zip the dataset only
bash scripts/train_fixed.sh --skip-pack        # data/packed/ already built
bash scripts/train_fixed.sh --datasets IIRS,M3
```

Once `data/packed/` exists, `scripts/train.sh` does steps 4/5 on its own:

```bash
bash scripts/train.sh --all --overwrite --epochs 30
```

### `--overwrite`

Without it, `train.sh` and `hpc_pbs_job.pbs` **skip** any slot whose checkpoint
already exists — and that skip prints to stdout only, never to Telegram. A
Telegram transcript therefore cannot distinguish "skipped" from "crashed", which
is how a previous grid appeared to lose slots that had merely been skipped. The
summary block now always lists the skipped runs explicitly.

```bash
bash scripts/train.sh --all --overwrite       # or: OVERWRITE=1 bash scripts/train.sh --all
bash scripts/hpc_launch.sh --overwrite        # threads OVERWRITE=1 through qsub -v
```

### Batch size — one number **per dataset**

Batch size is held constant across all 4 models × 2 loss regimes **within** a
dataset. That is the axis the ablation compares on, so nothing about a row's
result can be attributed to it.

It is deliberately **not** constant across datasets. There is no controlled
comparison between sensors to protect — they differ in band count, scene count,
spatial sampling and SNR — and these YAMLs already vary `vae_3d_base_ch` and
friends per dataset for parameter matching. Memory per sample varies ~2× between
sensors, so one global value would idle most of the card on the light ones.

It *does* have to hold across **platforms** for a given dataset, or within-dataset
fairness breaks the moment one slot runs on Kaggle and another on the lab. Each
number is therefore derived against the tightest budget any slot for that dataset
will see (the lab's 24 GB) and reused unchanged everywhere.

| Dataset | `batch_size` | Binding model | GB/sample | Predicted peak |
|---|---|---|---|---|
| IIRS   | **32** | `vae-3d-spatio-spectral` | 0.500 | 16.2 GB |
| M3     | **32** | `vae-1d-pixelwise` | 0.496 | 16.1 GB |
| AVIRIS | **16** | `vae-3d-spatio-spectral` | 0.880 | 14.3 GB |
| CRIMS  | **16** | `vae-3d-spatio-spectral` | 0.961 | 15.6 GB |

against 20.4 GB usable (24 GB less 15 % headroom), rounded down to a power of two
so the `nn.DataParallel` split stays even.

> **M3's binding model is `vae-1d-pixelwise`, not `vae-3d`.** Its MLP hidden dims
> are ~2,900–3,000 at *every* sensor, so its memory is essentially independent of
> band count (~0.50 GB/sample throughout) and it caps M3 long before the 3D model
> does. Scaling M3's batch from `base_ch × C` — which would have suggested 128 —
> is wrong for exactly that reason. Measure, don't extrapolate.

All four clear the other platforms: HPC's 40 GB has headroom, and Kaggle's 2 × T4
sees `batch_size/2` per 15 GB device (worst case CRIMS: 8 × 0.961 + 0.22 ≈ 7.9 GB).

```bash
# natively on the target GPU
PYTHONPATH=. python utils/find_max_batch.py --budget-gb 24 --time

# or from a smaller card: measures B=1,2,4 and fits peak = fixed + B*marginal
PYTHONPATH=. python utils/find_max_batch.py --budget-gb 24 --fit

# one global number across all sensors, if you ever want that instead
PYTHONPATH=. python utils/find_max_batch.py --budget-gb 24 --global
```

### Full options

```bash
python train/train.py --help
```

```
--data-root        Legacy per-patch tree; forces the legacy backend
--packed-root      Override the packed-shard dir (default: data/packed/<DS>)
--cache-ram        Load the whole split into RAM (removes disk from the loop)
--limit-train      Further cap training patches without re-packing
--num-workers      DataLoader workers   (default: 8)
--epochs           Training epochs      (default: 100)
--batch-size       Batch size           (default: 32)
--lr               Learning rate        (default: 1e-4)
--beta             KL weight            (default: 0.001)
--lambda-physics   SAM weight           (default: 0.5)
--log-recon-every  W&B recon log freq   (default: 10 epochs)
--ckpt-dir         Checkpoint directory (default: checkpoints/)
--wandb-project    W&B project name     (default: hsi-pi-vae)
--wandb-entity     W&B entity / team    (default: account from login)
--no-wandb         Disable W&B logging
```

### W&B metrics logged

| Metric | Description |
|--------|-------------|
| `train/loss` | Total loss per epoch |
| `train/mse` | Combined MSE (final + 0.5·spatial + 0.5·spectral) |
| `train/sam` | SAM physics prior loss |
| `train/kld` | Combined KL divergence |
| `train/lr` | Learning rate (cosine annealed) |
| `val/*` | Same metrics on the validation set |
| `reconstructions` | Original vs reconstructed patch pairs (1500 nm band) |

### Checkpoints

Saved to `--ckpt-dir/`:
- `best_model.pt` — best validation loss checkpoint
- `epoch_NNNN.pt` — periodic checkpoint every 10 epochs

Each checkpoint contains `epoch`, `model_state_dict`, `optimizer_state_dict`, `loss`.

---

## 6. Model Fix Notes

The original model had a **latent-dim mismatch** that prevented any forward pass:

| Branch | Problem | Fix applied |
|--------|---------|-------------|
| Spatial | `Encoder.linear → latent_dim`; `reparameterize` chunks → `latent_dim/2`; `Decoder.linear` expects `latent_dim` → shape mismatch | Encoder now emits `2*latent_dim`; Decoder `in_features=latent_dim` |
| Spectral | Same pattern along `dim=1` of the `(B, spectral_latent_dim, H, W)` map | Encoder emits `2*spectral_latent_dim` channels; Decoder `in_features=spectral_latent_dim` |

The `SpatialEncoderDecoder.forward` and `SpectralEncoderDecoder.forward` methods were also
updated to include reparameterization (they now return `z, mu, logvar, reconstruction`).

---

## 7. Caveats

> **compile-pass ≠ verified-run**

All `.py` files have been byte-compiled with `python -m py_compile` and all
bash scripts checked with `bash -n`. The dimension arithmetic has been verified
analytically. However, the pipeline has **not been end-to-end executed** (per
project instructions). Before your first full training run:

1. **Shape dry-run** — run a one-batch forward pass with dummy data to confirm
   all tensor shapes chain correctly:
   ```python
   import torch
   from train.train import HSI_DualStream_PI_VAE
   from utils.config import settings
   
   model = HSI_DualStream_PI_VAE(
       conv_output_c=settings.conv_output_c,
       conv_output_h=settings.conv_output_h,
       conv_output_w=settings.conv_output_w,
   )
   x = torch.randn(4, 64, 64, 108)
   out = model(x)
   print([o.shape for o in out])
   ```

2. **`wandb login`** — run once to store credentials; subsequent runs are silent.

3. **PYTHONPATH** — always run from the repo root with `PYTHONPATH=.` set
   (the bash scripts do this automatically). Running `python train/train.py`
   directly puts `train/` on `sys.path` and breaks all `from modules.*` /
   `from utils.*` imports.

4. **Memory** — each raw cube is ~3.6 GB. The preprocessing pipeline loads one
   cube at a time. Ensure ≥8 GB free RAM before running `scripts/preprocess.sh`.

5. **Pack before training.** The dataloader falls back to the legacy per-patch
   path when no shard exists and logs a WARNING; it works, but it is the slow
   path that made the previous grid disk-bound.

6. **Notebook config duplication.** `notebooks/*.ipynb` cell 2 inlines
   `utils/config.py` + `utils/hyperparams.py` + all four YAMLs so the notebooks
   run standalone on Kaggle. Any change to band counts, model widths, latent
   knobs, or batch sizes must be mirrored into **all four** notebooks. This
   duplication is exactly how `CRIMS: 544` survived in five places at once.

7. **`verify_channels` is packed-aware, and must stay that way.** A packed shard
   has already had `crop_bands` applied, so its band count is the *effective*
   count; a processed patch still carries the *raw* one. `probe_channels`
   returns `(count, source, location)` and callers must honour `source`.
   Conflating the two is what made all seven CRIMS slots fail with
   `on-disk patches have 456 bands but raw_channels says 457` — the data and the
   packing were both fine. `inspect_channels.py` and `verify_channels` now share
   the helper so they cannot disagree again (previously the inspector passed
   while every training slot failed).

---

## 8. HPC / IITD (PBS Pro)

The 28-run ablation grid targets IITD's Padum HPC (PBS Pro, A100 80GB
compute nodes). **Two separate hosts, two separate filesystems**:

- **login node** (`${HPC_USER}@${HPC_HOST}`) — reachable directly from the
  lab. No `qsub` here; it's a staging/jump host only.
- **compute node** (`${HPC_INNER_HOST}`, default alias `hpc`) — reached by
  `ssh`-ing into the login node and, from there, `ssh hpc`. `qsub`/`qstat`/
  `qdel` and the running jobs live here. Compute nodes have **no outbound
  internet**.

`scripts/hpc_common.sh` provides the two-hop primitives every other HPC
script builds on: `login_ssh "<cmd>"` (one hop), `compute_ssh "<cmd>"` (two
hops — base64-encodes the payload before the first hop so nested quoting in
`<cmd>` never has to survive two shell re-parses), `compute_rsync_push`, and
`resolve_hpc_roots` (fills `HPC_LOGIN_REPO_ROOT` / `HPC_COMPUTE_REPO_ROOT`
with back-compat fallback from `HPC_PROJECT_DIR`).

Consequences of the split filesystem:

- **Python env** — this repo ships a prebuilt `.venv/` (`USE_SHIPPED_VENV=1`,
  the default) that gets pushed lab→login→compute and used **as-is**;
  `scripts/hpc_bootstrap.sh` verifies it imports `torch`+`wandb` on the
  compute node rather than reinstalling anything. The old `pip download` /
  offline-wheels path (`USE_SHIPPED_VENV=0`) still exists as a fallback.
  **Invariant: never `source .venv/bin/activate`** anywhere in the HPC
  path — a venv rsynced from another machine has a dead `VIRTUAL_ENV` path
  baked into `bin/activate`, so activation silently no-ops and `python`
  resolves to the wrong interpreter. Always invoke `.venv/bin/python` by
  absolute path (it *is* relocatable — it derives its prefix from the
  adjacent `pyvenv.cfg`).
- **Rsync excludes must be anchored** (`/logs/`, `/model/`, `/data/`, …,
  with a leading `/`). An unanchored `--exclude='wandb/'` matches rsync's
  "final path component at any depth" rule and silently drops
  `.venv/lib/.../site-packages/wandb/` along with the repo's `wandb/` —
  `.venv/` gets its own rsync pass with only `__pycache__/`/`*.pyc` excluded.
- **wandb** — `WANDB_MODE=offline` on the compute node. The lab machine runs
  `wandb sync wandb/offline-run-*` after `hpc_pull_results.sh` completes.
- **Telegram** — `hpc_preflight.sh` probe 5 checks whether the compute node
  has direct outbound internet; if so, `utils/notify.py`'s tier-2 direct
  `sendMessage` handles everything with no tunnel. Otherwise, chained
  reverse tunnels: lab→login (`autossh`, background process) and
  login→compute (`autossh`, tmux session `prism_inner_tunnel` on the login
  node). The message forwarder (`scripts/notify_forwarder.py`) now runs
  **on the compute node** in tmux session `prism_forwarder` — that's where
  `logs/notify_queue.jsonl` actually gets written (CWD-relative in
  `utils/notify.py`, and jobs `cd` to the compute root) — and drains it
  through the chained tunnel to the lab's `utils/notify_relay.py`.
- **Result return** (compute→login→lab) — each PBS array element ends by
  calling `scripts/hpc_push_results.sh`, which **unconditionally** writes a
  `logs/pending_push/<idx>` marker on the compute node *before* attempting
  anything (so a fallback path always has ground truth), then — only if
  `PUSH_RESULTS_FROM_JOB=1` (set only when preflight probe 6 confirms
  compute→login connectivity) — tries to rsync its checkpoint/logs/wandb
  straight to the login node and clears the marker on success. Whether or
  not that push runs, `scripts/hpc_collector.sh` (tmux session
  `prism_collector` on the login node, started by `hpc_launch.sh`) polls
  the compute node for `pending_push/` markers on `COLLECTOR_INTERVAL`
  seconds and **pulls** anything still marked — pull, not push, because
  login→compute is the direction already proven to work. The lab-side
  `hpc_grid_watcher.sh` then pulls login→lab as before, now also fast-pathed
  by checking `logs/grid_done/<idx>` markers before falling back to qstat.

### Entry point

The junior runs two scripts on the lab machine — preflight first, always:

```bash
bash scripts/hpc_preflight.sh   # read-only: both hops, tools, shipped-venv sanity
bash scripts/hpc_launch.sh      # does the actual work
bash scripts/hpc_launch.sh --overwrite   # retrain slots that already have a ckpt
```

`--overwrite` (or `OVERWRITE=1`) is threaded through `qsub -v` into
`hpc_pbs_job.pbs`, which otherwise skips any slot whose checkpoint already
exists on the compute node.

**Staging the packed data.** `hpc_launch.sh` no longer rsyncs `data/` at all
(`--exclude='/data/'` covers `data/packed/` too) — the dataset is staged onto the
compute node manually, once. Copy `data/packed/` there by whatever route is
convenient; note that `hpc_common.sh:count_npy_remote` counts `*.npy` files and
the packed layout is 3 files per dataset rather than ~15,000, so do not use that
count as a completeness check for packed data.

`hpc_launch.sh` sanity-checks the WiFi (`mlr lab 5g`) and both SSH hops,
skips the wheel build (shipped-venv default), independently rsyncs repo /
`.venv/` / `data/processed/` lab→login then login→compute (each of the
three skipped if already present+intact on the target), runs
`hpc_bootstrap.sh` **on the compute node** via `compute_ssh`, brings up the
Telegram chain (§ above), and `qsub`'s the smoke + full array jobs via
`compute_ssh`. `scripts/hpc_train.sh` is now a thin wrapper —
`exec bash scripts/hpc_launch.sh --resume "$@"` — kept for anyone with the
old command memorized; the "skip what's already staged" behaviour it used
to duplicate is now native to `hpc_launch.sh` itself.

Config lives in `scripts/hpc_config.env` (copied from
`.example`) — every runtime-fill value is marked `FILL_ME` with an
"how to obtain" hint in `docs/hpc_wiki.md`. Full walkthrough for the
junior: **`docs/hpc_wiki.md`**.

### Grid manifest

The 28 grid slots are defined **once** in `scripts/grid_manifest.sh` and
sourced by both `scripts/train.sh --all` and `scripts/hpc_pbs_job.pbs`.
Slot ↔ tuple mapping:

- slots 1–7:   IIRS   (vae-our + 3 baselines × standard/physics)
- slots 8–14:  M3
- slots 15–21: AVIRIS
- slots 22–28: CRIMS

The PBS script picks its slot from `${PBS_ARRAY_INDEX}` via `grid_lookup`.

### Notification format (`utils/notify.py`)

`RunNotifier` emits three message types:

- **[START]** — once at run start, with resolved hyper-params in a `<pre>` block.
- **[HB]** — every `log_every` epochs (default 10), with current-epoch
  train/val loss/MSE/SAM/KLD, wall time, ETA, and best-so-far pointer.
- **[OK] / [FAIL] / [STOP]** — once at run end, with a stride-10 table and
  (on failure) the last ~60 log lines + exception traceback.

### Perf changes (math-preserving)

`train/train.py` now:

- Auto-picks BF16 autocast on Ampere+ (SM8.0+), FP16 + `GradScaler` elsewhere.
- Applies `channels_last` (2D models) / `channels_last_3d` (`vae-3d-spatio-spectral`).
- Accumulates loss/MSE/SAM/KLD on-GPU each step; one `.item()` per epoch.
- Updates the tqdm postfix every 20 steps, not every step.
- Enables TF32 for FP32 matmul paths + `cudnn.benchmark = True`.
- Wraps in `nn.DataParallel` only when `--allow-multi-gpu` is passed AND >1 GPU
  is visible (default single-GPU on HPC).
- Optional `torch.compile(model, mode="reduce-overhead")` via `--compile`,
  auto-skipped for `vae-3d-spatio-spectral` where Dynamo trips.
- DEBUG log level for the first `--debug-epochs` (default 3) epochs, then INFO.
- Shape-only tensor logging via `utils.logging_setup.log_tensor` — never
  dumps values.

`utils/training/dataloader.py`:

- `persistent_workers=True` (drops the spawn-context re-fork cost).
- `prefetch_factor=4`.
- Per-patch max cached in `__init__` from the sidecar `manifest.json`
  (written by `utils/dataset/slice.py`) — no per-item `.max()` scan.

### Compile-pass caveat for new bash / .pbs

All bash scripts (`hpc_common.sh`, `hpc_preflight.sh`, `hpc_launch.sh`,
`hpc_bootstrap.sh`, `hpc_pbs_job.pbs`, `hpc_push_results.sh`,
`hpc_collector.sh`, `hpc_smoke_watcher.sh`, `hpc_grid_watcher.sh`,
`hpc_pull_results.sh`, `hpc_train.sh`, `grid_manifest.sh`) and Python files
(`notify_relay.py`, `notify_forwarder.py`, updated `train.py` / `notify.py` /
`logging_setup.py` / `dataloader.py` / `slice.py`) have been passed through
`python -m py_compile` and `bash -n`. The two-hop `login_ssh`/`compute_ssh`
plumbing in `hpc_common.sh` was additionally exercised against a local mock
SSH chain (verifying the base64 payload round-trips correctly through the
login→compute hop) and `hpc_launch.sh --dry-run` / `hpc_preflight.sh` were
run end-to-end against that mock without a crash. **None of this has been
executed against the real IITD cluster.** The wiki's smoke-test flow
(`HPC_ARRAY_RANGE=1-1`, i.e. what `hpc_launch.sh` submits first) is the
recommended first real run — and `bash scripts/hpc_preflight.sh` should be
run before it every time, since it's the one check that can only be
answered against the real cluster (whether the shipped `.venv`'s
interpreter actually exists on the compute node, whether the two SSH hops
are passwordless, etc).

## 10. Performance — why the grid was slow

Measured from the 5-epoch smoke run (2026-08-19) and analytic MAC counts. Kept
here so nobody has to re-derive it.

### The numbers that started this

| Model | IIRS wall (5 ep) | GMAC/sample |
|---|---|---|
| vae-3d-spatio-spectral | 2h31m | 2,301 |
| vae-our | 1h14m | 467 |
| vae-1d-pixelwise | 39m | 101 |
| vae-standard | 40m | **11** |

`vae-standard` needs **40× fewer FLOPs** than `vae-our` and took the same 40
minutes. That single comparison says most of it: the grid was not
compute-bound.

### Three independent bottlenecks

**(a) Disk I/O — the real ceiling for 3 of 4 models.** One IIRS epoch opened
14,624 + 3,084 separate `.npy` files totalling ~74 GB, off the external drive at
`/media/yashdeep/New Volume 21/…`. 74 GB ÷ 8.1 min ≈ 152 MB/s — exactly
external-drive speed, and a hard floor of ~8 min/epoch *whatever model runs*.
(Local NVMe measures 726 MB/s on the same patches.)
→ Fixed by `utils/dataset/pack.py` (§4 Step 2) + the patch cap.

**(b) A full dataset re-read before every run.** `dataloader.py` fell back to
`np.load(p).max()` over every patch when `manifest.json` was absent — and it was
absent for every dataset. Another ~74 GB read, in the main process, at the start
of each of the 28 runs, *before* `RunNotifier` existed, so it never appeared in
any log. ≈10 min × 28 runs ≈ 4.5 h of pure waste.
→ Fixed: normalisation is applied once at pack time.

**(c) Two architectural FLOP sinks.**

*`vae-our`'s spectral branch was 99.7% of its own cost.* `SpectralBranch.Encoder`
set `out_c = settings.input_channels`, making the Conv1d blocks `1 → C → 2C`
channels — applied to **4,096 independent pixel spectra per patch**. At AVIRIS
that is a 424→848-channel convolution. Width was pinned to band count for no
principled reason, which also swung `vae-our`'s parameter count 3× across
sensors (24.5 M at IIRS vs 73 M at CRIMS) and quietly undermined the cross-sensor
comparison. Decoupling it (`spectral_base_ch = 32`) is a three-line change and
leaves the structure — per-pixel 1D convs, stride-2 halving, channel doubling,
linear → spectral latent, symmetric decoder, late fusion, MSE+βKLD+λSAM —
completely intact.

*`vae-3d` kept full spectral depth through every block.* `_DOWN_S = (1, 2, 2)`,
so the decoder's final `ConvTranspose3d` ran at `C × 64 × 64` (~500 G of the
AVIRIS total on its own). Now `(2, 2, 2)` with `_DOWN_K = (4, 4, 4)`.

> The kernel change is **required**, not cosmetic. With depth `k=3, s=2, p=1`
> the forward gives `ceil(L/2)` but the transpose gives `2L − 1`, compounding
> over three blocks to `53 → 105 → 209 → 417` at AVIRIS — short of 424, so no
> crop can recover it. `k=4, s=2, p=1` halves and doubles exactly.

CLAUDE.md previously described stride-1 depth as a *"design choice for
robustness"* — i.e. arithmetic convenience for arbitrary `C`. The baseline's
actual stated hypothesis is that 3D kernels inevitably average neighbouring
bands together with neighbouring pixels; striding the depth axis strengthens
that, and the pad/crop keeps the round-trip exact for every sensor.

### Result

| | IIRS | M3 | AVIRIS | CRIMS |
|---|---|---|---|---|
| vae-our GMAC/sample **before** | 467 | 21 | 2,021 | 2,512 |
| vae-our GMAC/sample **after** | **14.3** | **5.0** | **23.9** | **25.8** |
| vae-3d GMAC/sample **before** | 2,301 | 362 | 7,576 | 12,114 |
| vae-3d GMAC/sample **after** (param-matched) | ~459 | ~130 | ~825 | ~923 |
| vae-our params **before** | 24.5 M | 12.2 M | 48.5 M | 73.0 M |
| vae-our params **after** | 12.4 M | 11.2 M | 13.7 M | 13.9 M |

`vae-our` drops out of the critical path entirely and becomes I/O-bound like the
baselines. `vae-3d` remains the long pole.

**Estimated** at the lab GPU's back-derived throughput (~111 TFLOPS effective,
from dividing the analytic MAC count for the old `vae-3d` by its observed smoke
wall time) and 7,000 patches/epoch, `vae-3d` lands around 3 min/epoch at IIRS and
5 min at AVIRIS — 1.5–3 h for 30 epochs, inside the target with margin.

Treat that as an estimate, not a measurement. The analytic 3D MAC count was ~2.4×
higher than the observed wall time implied (predicted ~6 h, observed 2h31m), and
the 111 TFLOPS figure absorbs that error, so the projection inherits it. The
error direction is safe — if the analytic count was high, the real runs are
*faster* than stated. **The smoke pass gives the true number two epochs in:**
`grep 'wall ' logs/train_*.log`.

### Why CRIMS produced nothing at all

Two config bugs, plus one reason you never saw them:

- `processed_root` was `data/processed/crims` (lowercase); the lab directory is
  `CRIMS`. Linux is case-sensitive → `FileNotFoundError`.
- `input_channels` said **544**; the patches are **457** (verified across all 15
  scenes). 457 is prime, so the spectral round-trip can never be exact
  (`457 // 4 = 114 → 456 ≠ 457`). Now cropped to 456, exactly as M3 is cropped
  85 → 84.
- **The silence was its own bug.** `RunNotifier` was constructed *after* the
  dataloaders and model were built, and the `try/except` that emits `[FAIL]`
  opened later still. Anything throwing during setup sent zero Telegram output.
  `RunNotifier` + `send_start()` now run **before** any disk access, and the
  `try` covers dataloader and model construction, so setup failures report with
  a traceback instead of vanishing.

### Still open — not addressed here

`sam = 1.5708` (**exactly π/2**, i.e. reconstruction orthogonal to input —
decoder collapsed toward a constant) appears in three smoke runs:
`vae-standard|IIRS|standard`, `vae-3d|IIRS|standard`, `vae-3d|AVIRIS|standard`.
Only on `standard` runs — no SAM term to prevent it — with
`mse ≈ 0.0030 ≈ mean(x²)`, which is consistent with collapse. Five epochs is too
few to call it, but if it persists at 30 those baselines produce nothing usable
however fast they run, and the ~4× batch-size increase interacts with it. Worth
checking after the first full grid.

---

## 11. Experimental controls — rate and parameters

The ablation controls **two independent resources**. Conflating them was the
single biggest methodological hole in the earlier grid.

| Resource | What it bounds | Knob | Tool |
|---|---|---|---|
| **Parameters** | how complex a mapping can be learned | `*_base_ch`, `vae_1d_hidden_dims` | `utils/check-model-params.py --solve` |
| **Latent rate** | how much information can pass the bottleneck | `spectral_latent_dim`, `vae_*_latent_ch/dim` | `utils/match_latent_rate.py --ratio 64` |

Only parameters were being matched. Rate floated **512×** within a dataset:

| dataset | input | vae-standard | vae-3d | vae-1d | vae-our |
|---|---|---|---|---|---|
| IIRS | 1,048,576 | 1,024 (1024:1) | 16,384 (64:1) | 131,072 (8:1) | 524,544 (**2:1**) |
| M3 | 344,064 | 1,024 (336:1) | 5,632 (61:1) | 131,072 (3:1) | 524,544 (**1.52× the input**) |
| AVIRIS | 1,736,704 | 1,024 (1696:1) | 27,136 (64:1) | 131,072 (13:1) | 524,544 (**3:1**) |
| CRIMS | 1,867,776 | 1,024 (1824:1) | 29,184 (64:1) | 131,072 (14:1) | 524,544 (**4:1**) |

For an autoencoder, reconstruction quality is bounded by rate almost by
definition — a 2:1 bottleneck beats a 1024:1 one regardless of what is inside
it. So rate was the *more* important of the two controls to have been missing.
And on M3 the "latent" was 1.52× larger than the cube it encoded: an
over-complete code that can copy the input outright, and unusable as an LDM
backbone (Stable Diffusion's AutoencoderKL is 48:1).

### The 64:1 target

`vae-3d` is already exactly 64:1 (8×8×8 downsampling × 8 latent channels), so it
is **unchanged** and cannot be accused of being re-tuned for the comparison.

| dataset | target | `spectral_latent_dim` | `vae_standard_latent_ch` | `vae_3d_latent_ch` | `vae_1d_latent_dim` |
|---|---|---|---|---|---|
| IIRS | 16,384 | 128 → **4** (+1.6 %) | 16 → **256** (exact) | **8** (unchanged) | 32 → **4** (exact) |
| M3 | 5,376 | 128 → **1** (−19 %) | 16 → **84** (exact) | **8** (+4.8 %) | 32 → **1** (−23.8 %) |
| AVIRIS | 27,136 | 128 → **7** (+6.6 %) | 16 → **424** (exact) | **8** (unchanged) | 32 → **7** (+5.7 %) |
| CRIMS | 29,184 | 128 → **7** (−0.9 %) | 16 → **456** (exact) | **8** (unchanged) | 32 → **7** (−1.8 %) |

Exact 4-way matching is only possible on IIRS (C is a power of two). The
per-pixel models can only hit multiples of 4,096 elements and 4,096 does not
divide M3's 5,376 target, so M3 is the loosest at −24 % — inside the ±25 %
tolerance, but only just. Achieved rates are recorded per cell and reported;
P2 regresses on the achieved value, never the target.

**What is deliberately NOT matched: latent shape.** `vae-standard` is a spatial
grid with no spectral axis; `vae-our` is a global vector beside a
full-resolution per-pixel spectral map. That geometry *is* the architecture
under test — forcing a common shape would destroy the thing being measured.
Only the scalar count matches. `vae-our`'s spectral latent drops 128 → 4
channels per pixel, which is severe, but the shape survives.

```bash
python utils/match_latent_rate.py --ratio 64            # solve + paste-ready YAML
python utils/match_latent_rate.py --ratio 64 --check    # verify, non-zero on failure
python utils/match_latent_rate.py --ratio 64 --fit      # (find_max_batch) from a small GPU
```

`match_latent_rate.py` cross-checks its closed forms against the real models on
every run, so a drift between it and `modules/` is caught rather than shipped.

---

## 12. Falsification suite

`inference/probes.py` — seven probes on **frozen** checkpoints, each answering a
specific way the headline result could be fake. Thresholds live in
`inference/preregistration.yaml`, are fixed **before** any run, and the module
refuses to start without that file. Reasoning for every number:
`docs/preregistration.md`.

| Probe | Question | Fails when |
|---|---|---|
| **P1** trivial floors | better than predicting the mean? | below global/region/fold/patch mean or 1000 random draws |
| **P2** latent rate | capacity or architecture? | rate outside ±25 % of 64:1 |
| **P3** collapse | is the latent used? | <1 % active units, or latent-swap moves SAM <2 % |
| **P4** spatial shuffle | uses spatial context? | SRI < 0.02 (i.e. pixelwise in disguise) |
| **P5** band inpainting | has a spectral prior? | <10 % gain over mean-fill on masked bands |
| **P6** purification | does it actually denoise? | NPR ≥ 0.9 (passes input noise through) |
| **P7** linear probe | latent = chemistry or nuisance? | physics R² < 0.5 |

Plus `inference/stats.py`: paired bootstrap CIs, paired permutation tests,
Holm–Bonferroni within each dataset's model-pair family, and Cliff's delta.

### Three things the suite corrects for

1. **SAM has a non-zero floor.** A *perfect copy* scores 0.0223 on IIRS, not 0
   (the epsilon in its norm). The identity oracle is the real ceiling, so the
   suite reports *headroom captured*, not raw scores.
2. **SAM has a π/2 contamination.** A pixel with spectral energy below that
   epsilon contributes **exactly π/2 whatever the model predicted**. CRIMS has
   ~24 % such pixels → a hard raw-SAM floor near 0.377 rad unrelated to model
   quality. Use `sam_valid`, which excludes them.
3. **Significance is not evidence.** At n = 3,084 a 0.05 dB gap gives p = 0.0005.
   Every comparison carries a preregistered minimum effect (0.5 dB / 0.005 rad /
   0.01 SSIM); anything below is labelled `significant_but_negligible` and is
   **not** a win.

### Probe self-test

`vae-1d-pixelwise` is *exactly* permutation-equivariant, so in P4 its shuffled
and intact scores must match to 1e-6. A larger deviation is reported as
`PROBE_BUG` — the probe is wrong, not the model. Check this before believing any
P4 result.

### Running it

```bash
bash scripts/inference.sh                      # recon + probes + downstream + verdict
bash scripts/inference.sh --probes-only        # just the suite
bash scripts/inference.sh --datasets CRIMS
bash scripts/inference.sh --max-patches 0      # whole split instead of 512
```

Outputs:

```
results/VERDICT.txt   <- read this one: why each model wins or loses, per dataset
results/probes.csv       per-cell probe metrics + PASS/FAIL/INVALID
results/stats.csv        pairwise deltas, CIs, p, Holm, effect sizes
results/probes/*.json    raw per-cell output
```

A claimed win requires: cell `VALID` **and** rate matched **and** the pairwise
difference significant after Holm **and** above the effect floor.

---

## 13. Phase 2 (Future)

The VAE encoder (`SpatialEncoderDecoder`, `SpectralEncoderDecoder`) will serve as
the backbone for a Latent Diffusion Model (LDM) that performs diffusion-based
purification of the compressed latent representations from satellite imagery.
The `standalone forward` methods on both encoder-decoder classes expose the
full `(z, mu, logvar, reconstruction)` return for easy LDM integration.
