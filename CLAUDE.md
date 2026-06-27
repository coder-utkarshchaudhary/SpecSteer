# Pipeline Status — Dual-Stream Physics-Informed VAE for HSI

> **Last updated:** 2026-06-21  
> **Status:** All components implemented and byte-compiled.  
> ⚠️ Compile-pass ≠ verified-run. See [Caveats](#caveats) before running.

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
               │  Conv2D × 4 (spatial↓)  flatten → z_p  │
               │  flatten → z_s                          │
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
| `utils/training/dataloader.py` | HSIPatchDataset + DataLoader factory | ✅ New |
| `modules/SpatialBranch.py` | Spatial encoder-decoder | ✅ Fixed |
| `modules/SpectralBranch.py` | Spectral encoder-decoder | ✅ Fixed |
| `train/train.py` | Training loop, wandb, CLI, checkpointing | ✅ Rewritten |
| `scripts/preprocess.sh` | One-command preprocessing runner | ✅ New |
| `scripts/train.sh` | One-command training runner | ✅ New |
| `docs/file_processing.py` | Reference script (do not modify) | — |

---

## 3. Configuration (`utils/config.py`)

All settings live in the `Settings` dataclass. Defaults are set for 64×64 patches
with 108-band IIRS cubes. Change values in the dataclass; derived fields
(`conv_output_*`, `spectral_*`) are recomputed automatically in `__post_init__`.

### Key config values

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `input_height/width` | 64 | patch spatial size |
| `input_channels` | 108 | bands 7:115 of IIRS |
| `band_start / band_end` | 7 / 115 | reflective window |
| `norm_band_idx` | 41 | ≈1500 nm reference band (within selected subset) |
| `savgol_window` | 7 | Savitzky-Golay window length |
| `patch_size / patch_stride` | 64 / 48 | 25% overlap |
| `split_ratios` | (0.70, 0.15, 0.15) | train/valid/test |
| `reduced_dims` | 32 | spatial Conv1D output channels |
| `latent_dim` | 256 | spatial latent (post-reparameterize) |
| `n_2D_conv_blocks` | 4 | spatial bottleneck: 64→4 px |
| `spectral_n_1D_conv_blocks` | 2 | spectral bottleneck |
| `spectral_latent_dim` | 128 | per-pixel spectral latent |
| `spectral_transpose_c/l` | 216 / 27 | decoder reshape target |
| `spectral_linear_expansion_dim` | 5832 | 216 × 27 |

### Spectral dimension arithmetic

With `spectral_conv1D_kernel_size=4`, `stride=2`, `padding=1`:
```
Conv1d:          L_out = L_in // 2
ConvTranspose1d: L_out = 2 * L_in

Encoder:  108 → 54 → 27  (L);  channels  1 → 108 → 216
Decoder:   27 → 54 → 108 (L);  channels 216 → 108 → 1
```

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

---

## 5. Training

### Prerequisites

```bash
# Install dependencies (already in .venv):
pip install torch torchvision scipy wandb

# One-time W&B login (run once; stores credentials in ~/.netrc):
wandb login
```

### Quick start

```bash
bash scripts/train.sh --epochs 100
```

### Full options

```bash
python train/train.py --help
```

```
--data-root        Processed data root  (default: data/processed)
--num-workers      DataLoader workers   (default: 4)
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

---

## 8. Phase 2 (Future)

The VAE encoder (`SpatialEncoderDecoder`, `SpectralEncoderDecoder`) will serve as
the backbone for a Latent Diffusion Model (LDM) that performs diffusion-based
purification of the compressed latent representations from satellite imagery.
The `standalone forward` methods on both encoder-decoder classes expose the
full `(z, mu, logvar, reconstruction)` return for easy LDM integration.
