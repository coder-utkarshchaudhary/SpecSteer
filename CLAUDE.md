# Pipeline Status — Dual-Stream Physics-Informed VAE for HSI

> **Last updated:** 2026-08-16  
> **Status:** All 4 ablation models + downstream experiments implemented and byte-compiled.
> IITD HPC (PBS Pro) launcher + reverse-tunnel Telegram relay added.
> Training loop performance-tuned (BF16 on Ampere+, channels_last, per-epoch metric sync).  
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
| `modules/vae_our.py` | vae-our dual-stream PI-VAE (+ encode/decode_latents) | ✅ |
| `modules/vae_standard.py` | Baseline A: 2D spatial VAE (AutoencoderKL-style) | ✅ New |
| `modules/vae_3d.py` | Baseline B: 3D spatio-spectral VAE | ✅ New |
| `modules/vae_1d.py` | Baseline C: 1D pixelwise VAE | ✅ New |
| `modules/losses.py` | Shared SAM + KL loss primitives | ✅ |
| `modules/registry.py` | CLI name → model class; model contract | ✅ |
| `train/train.py` | Training loop, wandb, CLI, checkpointing | ✅ Rewritten |
| `inference/inference.py` | Reconstruction eval (MSE/SAM/PSNR/SSIM) | ✅ |
| `inference/downstream.py` | Latent noise-injection + interpolation experiments | ✅ New |
| `notebooks/*.ipynb` | Per-model self-contained training notebooks | ✅ |
| `inference/notebooks/*.ipynb` | Per-model self-contained eval notebooks | ✅ |
| `scripts/preprocess.sh` | One-command preprocessing runner | ✅ New |
| `scripts/train.sh` | Training runner (single run or full 21-run grid) | ✅ New |
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

| Model | Registry name | Latent | Hypothesis |
|-------|---------------|--------|------------|
| A: 2D Spatial | `vae-standard` | `(B, 16, 8, 8)` map | 2D convs blur pixel spectra → good PSNR, poor SAM |
| B: 3D Spatio-Spectral | `vae-3d-spatio-spectral` | `(B, 8, C, 8, 8)` volume | averages bands+pixels, param-heavy, collapse-prone |
| C: 1D Pixelwise | `vae-1d-pixelwise` | `(B, H, W, 32)` per-pixel | great chemistry (SAM), no spatial denoise → poor PSNR/SSIM |
| Proposed: PRISM | `vae-our` | `[(B,256), (B,128,H,W)]` | spatial+spectral isolation → high PSNR *and* low SAM |

Baselines A/B preserve the spatial grid at 8×8 (H,W ÷ 8) and reconstruct
exactly for any band count C (B keeps spectral depth at stride 1; A/C are
C-agnostic by construction), so all three run unchanged on IIRS/M3/AVIRIS.

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
```

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

## 9. Phase 2 (Future)

The VAE encoder (`SpatialEncoderDecoder`, `SpectralEncoderDecoder`) will serve as
the backbone for a Latent Diffusion Model (LDM) that performs diffusion-based
purification of the compressed latent representations from satellite imagery.
The `standalone forward` methods on both encoder-decoder classes expose the
full `(z, mu, logvar, reconstruction)` return for easy LDM integration.
