# PRISM: A Dual-Stream Physics-Informed VAE for Decoupled Representational Learning of Hyperspectral Images

**Paper under double-blind review at ICLR 2027.**

---

## Overview

Hyperspectral imaging (HSI) sensors record hundreds of contiguous spectral bands per pixel, encoding the reflectance signature of surface materials. Sensor degradation, transmission dropouts, and instrument noise routinely corrupt these cubes, motivating robust representation learning that can both compress and probabilistically repair corrupted spectra.

Conventional convolutional VAEs treat HSI cubes as thick images, applying 2D convolutions that mix spatial neighborhoods and spectral channels indiscriminately. This blurs the fine spectral signatures that carry chemical meaning and, under the tight bottleneck a VAE requires, induces posterior collapse or physically implausible spectral hallucinations.

**PRISM** (Physics-Informed Representation for Isolated Spectral-Spatial Modeling) resolves this conflict by routing spatial and spectral information through two independently-supervised encoder–decoder branches. Their reconstructions re-couple only at a final adaptive fusion step constrained by a differentiable Spectral Angle Mapper (SAM) penalty, which penalizes angular distortion in spectral space directly — the failure mode that naive fusion would otherwise be free to produce.

> Across three planetary and terrestrial HSI datasets and three baseline architectures (2D spatial, 1D pixelwise, 3D spatio-spectral), PRISM achieves **+6.949 dB PSNR** and **−0.112° SAM** over the strongest rival at a matched latent-rate bottleneck.

---

## Architecture

<!-- Figures will be placed here in the camera-ready version. -->
<!-- ![PRISM pipeline](figures/architecture_v1.png) -->
> **Figure 1 placeholder** — *Pipeline of PRISM for representation learning of HSI.* The corresponding figures are included as `figures/architecture_v1.png` and `figures/architecture_v2.png` in the LaTeX source (`paper/iclr/v1/`).

<!-- ![PRISM network structure](figures/architecture_v2.png) -->
> **Figure 2 placeholder** — *Network structure of PRISM.*

### Spatial stream

Responsible for preserving and denoising spatial context. The encoder applies a 3×3 convolution projecting the C-band input to 64 channels, followed by three strided (s=2) convolutional blocks halving the patch resolution from 64×64 down to 8×8. A final pointwise convolution outputs the (μ, log σ) pair of the spatial latent **z**_s ∈ ℝ^{d_s × 8 × 8} — an addressable spatial map rather than a global vector, so spatial texture and contrast can vary across the patch. The decoder mirrors this path with transposed convolutions.

### Spectral stream

Responsible for per-pixel chemical information. It never mixes information across pixels. The encoder applies two strided (s=2) 1D convolutions along the spectral axis (channels 1 → 32 → 64), then flattens and projects to a 2d_p-dimensional (μ, log σ) vector per pixel. These are reshaped back onto the spatial grid, giving a per-pixel latent map **z**_p ∈ ℝ^{d_p × H × W}. The decoder mirrors with two 1D transposed convolutions. Every layer operates on one pixel at a time, architecturally preventing cross-pixel spectral collusion.

### Adaptive gated fusion

Both streams produce a full reconstruction independently. A small two-layer 3×3 convolutional network learns a per-pixel, per-band gate α ∈ (0, 1) from the concatenated reconstructions:

```
x̂_f = α ⊙ x̂_s + (1 − α) ⊙ x̂_p
```

The gate is trained jointly under the SAM term, so it cannot reduce reconstruction loss by producing a spectrum that is spatially plausible but chemically hallucinated.

### Training objective

```
L = L_rec + β · L_KL + λ_phys · L_SAM
```

with β = 10⁻³ and λ_phys = 0.3, constant across all models and datasets (no annealing). `L_rec` is a normalized weighted sum of the fused and per-branch auxiliary reconstruction losses (5:1:1 ratio). `L_SAM` is the Spectral Angle Mapper applied only to the fused reconstruction.

---

## Repository Structure

```
SpecSteer/
├── modules/                     VAE architectures and shared utilities
│   ├── vae_our.py               PRISM (proposed model)
│   ├── vae_our_variants.py      PRISM-NL ablation variant
│   ├── vae_standard.py          2D Spatial VAE baseline
│   ├── vae_3d.py                3D Spatio-Spectral VAE baseline
│   ├── vae_1d.py                1D Pixelwise VAE baseline
│   ├── SpatialBranch.py         Spatial encoder/decoder
│   ├── SpectralBranch.py        Spectral encoder/decoder
│   ├── registry.py              Model registry and checkpoint resolution
│   ├── losses.py                SAM and KL divergence primitives
│   └── metrics.py               PSNR/SSIM (single unified implementation)
│
├── train/
│   └── train.py                 Model-agnostic training loop
│
├── inference/
│   ├── inference.py             Reconstruction evaluation
│   ├── probes.py                Falsification suite (5 diagnostic probes)
│   ├── downstream.py            Latent noise robustness + interpolation smoothness
│   ├── stats.py                 Bootstrap CIs, permutation tests, Holm–Bonferroni
│   ├── verdict.py               Per-dataset verdict synthesis
│   ├── aggregate.py             Multi-run aggregation → ablation_table.csv
│   └── preregistration.yaml     Preregistered probe thresholds
│
├── utils/
│   ├── config.py                Settings dataclass and per-dataset configuration
│   ├── hyperparams.py           YAML hyperparameter loading
│   ├── dataset/                 Preprocessing, slicing, and packing pipeline
│   │   ├── preprocess.py        Band selection, normalisation, Savitzky-Golay smoothing
│   │   ├── slice.py             Region-disjoint 70/15/15 split, 64×64 patch extraction
│   │   └── pack.py              fp16 memory-mapped shards with per-patch normalisation
│   └── hyperparam_configs/      Per-dataset YAML hyperparameter files
│
├── scripts/
│   ├── run_clean_grid.sh        Main entry point: full 5×4×2 ablation grid
│   ├── inference.sh             End-to-end evaluation (recon + probes + downstream + verdict)
│   ├── hpc_launch.sh            Two-hop HPC orchestration (PBS Pro)
│   └── ...                      Supporting launchers and watchers
│
├── notebooks/                   Self-contained Kaggle notebooks (one per model)
│   ├── vae-our.ipynb
│   ├── vae-standard.ipynb
│   ├── vae-3d-spatio-spectral.ipynb
│   └── vae-1d-pixelwise.ipynb
│
├── paper/iclr/v1/               LaTeX source (under review)
├── docs/
│   ├── preregistration.md       Rationale for every probe threshold
│   └── hpc_wiki.md              HPC reproducibility guide
│
├── requirements.txt
└── LICENSE
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: PyTorch ≥ 2.1, torchvision, numpy, scipy, scikit-image, wandb, pyyaml, tqdm.

A GPU with at least 16 GB VRAM is recommended for the full training grid. The `inference_smoke.sh` script runs a synthetic CPU smoke test (~15 min) to verify the evaluation pipeline before committing to a full GPU sweep.

---

## Datasets

| Dataset | Instrument | Mission | Bands | Wavelength range | Train / Val / Test patches |
|---|---|---|---|---|---|
| IIRS | Chandrayaan-2 IIRS | Lunar orbiter | 256 | ~0.8–5.0 μm | 14,624 / 3,084 / 3,084 |
| AVIRIS | AVIRIS | Airborne (terrestrial) | 424 | 0.4–2.5 μm | 11,027 / 2,180 / 1,950 |
| CRISM | MRO CRISM | Mars orbiter | 456 | 0.362–3.92 μm | 2,561 / 369 / 369 |

Band counts are the number of contiguous reflective channels retained after removing non-reflective or degenerate detector channels. Patches are 64×64 pixels extracted at stride 48 (25% overlap) from region-disjoint spatial splits of each source scene, ensuring no training patch overlaps any test patch.

### Data preprocessing

Raw HSI cubes are not included in this repository. After obtaining the source data, run the three-step preprocessing pipeline:

```bash
# Step 1 — Band selection, normalisation, Savitzky-Golay smoothing
python utils/dataset/preprocess.py --dataset IIRS --data-root /path/to/raw/data

# Step 2 — Region-disjoint split and 64×64 patch extraction
python utils/dataset/slice.py --dataset IIRS --data-root /path/to/data

# Step 3 — Pack patches into fp16 memory-mapped shards
python utils/dataset/pack.py --dataset IIRS --data-root /path/to/data
```

Repeat for each dataset (`IIRS`, `AVIRIS`, `CRISM`). Verify integrity with:

```bash
python utils/dataset/audit_pack.py --dataset IIRS --data-root /path/to/data
```

---

## Training

### Full ablation grid

The main entry point trains all five models on all four datasets at two random seeds:

```bash
DATA_ROOT=/path/to/data bash scripts/run_clean_grid.sh
```

This runs **5 models × 4 datasets × 2 seeds = 64 cells** sequentially: for each cell it trains, runs the full evaluation pipeline (reconstruction + probes + downstream), and then calls `verdict.py` and `aggregate.py` once per dataset. Cells are resumable via marker files in `results/.done/`.

### Single model

```bash
python train/train.py \
    --model vae-our \
    --dataset IIRS \
    --loss-type physics \
    --seed 67 \
    --data-root /path/to/data
```

Available `--model` values: `vae-our` (PRISM), `vae-our-nl` (PRISM-NL), `vae-standard`, `vae-3d-spatio-spectral`, `vae-1d-pixelwise`.

`--loss-type physics` trains with the full objective (MSE + β·KLD + λ·SAM); `--loss-type standard` ablates the SAM term.

Checkpoints are saved to `model/<dataset>/<stem>_seed<N>_best{sam,mse}.pt` — two checkpoints per cell, one selected by lowest validation SAM and one by lowest validation MSE.

### HPC (PBS Pro)

See `docs/hpc_wiki.md` for the full two-hop orchestration guide.

```bash
bash scripts/hpc_preflight.sh   # read-only checks first
bash scripts/hpc_launch.sh      # rsync, qsub smoke + full grid
```

---

## Evaluation

```bash
bash scripts/inference.sh --dataset IIRS --data-root /path/to/data
```

This runs four steps in sequence:
1. **Reconstruction evaluation** (`inference/inference.py`) — MSE, SAM-valid, PSNR, SSIM, SID, SCC, Q2^n on the held-out test split.
2. **Falsification suite** (`inference/probes.py`) — five diagnostic probes on every frozen checkpoint.
3. **Downstream readiness** (`inference/downstream.py`) — latent noise-injection robustness and chemical interpolation smoothness.
4. **Verdict + aggregation** (`inference/verdict.py`, `inference/aggregate.py`) — outputs `results/ablation_table.csv`.

For a quick synthetic CPU smoke test before the real sweep (~15 min):

```bash
bash scripts/inference_smoke.sh
```

---

## Reproducing Paper Results

All tables are generated from `results/ablation_table.csv` and the per-cell probe outputs in `results/probes/`. The correspondence is:

| Paper table | Output file | Generated by |
|---|---|---|
| Table 1 — Reconstruction quality | `results/ablation_table.csv` | `inference.py` + `aggregate.py` |
| Table 2 — Latent noise recovery | `results/downstream/noise_*.csv` | `downstream.py` |
| Table 3 — Chemical interpolation | `results/downstream/interp_*.csv` | `downstream.py` |
| Table 4 — Spectral band masking | `results/probes/*_p5_*.csv` | `probes.py` (Probe 5) |
| Diagnostics table | `results/probes/*_p{1..4}_*.csv` | `probes.py` (Probes 1–4) |

Run `bash scripts/inference.sh --dataset <DS>` for each dataset (`IIRS`, `AVIRIS`, `CRISM`) to populate all outputs.

---

## Falsification Suite

Five diagnostic probes run on every frozen checkpoint before any cross-model comparison is drawn. No probe imposes a pass/fail threshold — all results are diagnostic. A cell is excluded from reconstruction comparisons only if Probe 3 confirms posterior collapse.

| Probe | What it measures |
|---|---|
| **P1 — Trivial predictor floor** | PSNR lift over zero-rate predictors (global mean, fold mean, region mean) and over the per-patch spatial mean; headroom relative to an input-copy oracle. |
| **P2 — Latent rate audit** | Confirms every cell lands within ±25% of the preregistered common latent budget T (64:1 compression ratio target). |
| **P3 — Posterior collapse** | Active units (fraction of latent units with non-trivial KL divergence) and latent-swap Δ (swapping two patches' codes should change the reconstruction substantially). |
| **P4 — Spatial reliance shuffle** | Permutes the pixel grid while keeping each pixel's spectrum intact; the 1D Pixelwise VAE is the positive control (its SRI must be near zero, confirming the probe measures what it claims). |
| **P5 — Spectral band masking** | Zeros a contiguous 10% block at five positions spanning the spectrum; scores reconstruction on masked bands against a mean-fill baseline and a pass-through ratio. |

Thresholds are preregistered in `inference/preregistration.yaml` and documented with full rationale in `docs/preregistration.md`.

---

## Checkpoints

Training a full cell takes approximately 2–4 hours on an A100 GPU. Pretrained checkpoints will be released alongside the camera-ready version. To resume or run inference on an existing checkpoint:

```bash
python inference/inference.py \
    --model vae-our \
    --dataset IIRS \
    --select sam \
    --data-root /path/to/data \
    --checkpoint-dir /path/to/checkpoints
```

---

## Citation

```bibtex
[To be added upon publication.]
```

---

## License

This code is released under the [MIT License](LICENSE).
