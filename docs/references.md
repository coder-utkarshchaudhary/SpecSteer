# Baseline reference configurations — sources

> Retrieved via the Science Copilot paper search (Google Scholar provider) on
> 2026-09-04, during the capacity-protocol decision (docs/new_plan.md,
> "Capacity protocol"). These papers anchor the baseline widths in
> `utils/hyperparam_configs/hyperparam-config-*.yaml` and `utils/config.py`.
> The `paper/draft.md` bibliography should cite the canonical versions below;
> the retrieval URLs are Science Copilot redirect links kept for provenance.

---

## 1. vae-standard — AutoencoderKL widths (`vae_standard_base_ch: 128`, n_down 3)

**High-Resolution Image Synthesis with Latent Diffusion Models**
R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer.
*IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.*
Citations at retrieval: ~38,596.
Retrieval: <https://prod.api.sciencecopilot.modelfactory.amazon.dev/paper/redirect/dahwrJVseu>

**What it anchors.** The `vae-standard` baseline is an AutoencoderKL-style 2-D
conv VAE ("treats the cube as a thick RGB image"). Widths follow the LDM f8
autoencoder: base channel 128 with channel multipliers over 3 downsampling
stages (128/256/512). This makes vae-standard the largest model in the grid
(~23–24M params) — larger than PRISM, so it needs no capacity point.

## 2. vae-3d — 3D-CAE (`vae_3d_base_ch: 24`, REPRESENTATIVE)

**Unsupervised Spatial–Spectral Feature Learning by 3D Convolutional
Autoencoder for Hyperspectral Classification**
S. Mei, J. Ji, Y. Geng, Z. Zhang, X. Li, Q. Du.
*IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2019.*
Citations at retrieval: ~362.
Retrieval: <https://prod.api.sciencecopilot.modelfactory.amazon.dev/paper/redirect/UTLVr856u6>

Earlier workshop version (same line):
**Learning Sensor-Specific Features for Hyperspectral Images via 3-Dimensional
Convolutional Autoencoder** — J. Ji, S. Mei, J. Hou, X. Li, Q. Du.
*IEEE IGARSS, 2017.*
Retrieval: <https://prod.api.sciencecopilot.modelfactory.amazon.dev/paper/redirect/qROMrif6R4>

**What it anchors.** The `vae-3d` baseline is a fully-3D-conv VAE over the
(C, H, W) volume, per the 3D-CAE line. **`base_ch = 24` is a REPRESENTATIVE
default**: the exact per-layer filter counts were not recoverable from the
abstract/metadata at retrieval time (no accessible PDF), so a compact
tens-of-filters stack consistent with the paper's description was adopted.
**Swap in the exact counts and re-run if the PDF becomes accessible** — the
YAML comments carry the same caveat.

## 3. vae-1d — fully-connected spectral (V)AE (`vae_1d_hidden_dims: [512, 256, 128]`)

**Hyperspectral Unmixing Using a Neural Network Autoencoder**
B. Palsson, J. Sigurdsson, J. R. Sveinsson, M. O. Ulfarsson.
*IEEE Access, 2018.*
Citations at retrieval: ~315.
Retrieval: <https://prod.api.sciencecopilot.modelfactory.amazon.dev/paper/redirect/c1zX1INpie>

**Dual-Frequency Autoencoder for Anomaly Detection in Transformed
Hyperspectral Imagery**
Y. Liu, W. Xie, Y. Li, Z. Li, Q. Du.
*IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2022.*
Citations at retrieval: ~41.
Retrieval: <https://prod.api.sciencecopilot.modelfactory.amazon.dev/paper/redirect/pVB12r0O1d>

**What it anchors.** The `vae-1d` baseline is a per-pixel fully-connected VAE
on individual spectra — the standard architecture of the spectral autoencoder
line (Palsson et al. 2018 for unmixing; Liu et al. 2022 use four
fully-connected layers per coder for anomaly detection). Hidden widths
512/256/128 (4:2:1) are representative of this line at hundreds-of-bands
input. Note the contrast this replaces: the old param-matched widths
([2656–2748, ×2, ×1] solved to PRISM's 10.9M) were ~5× wider than anything in
these papers, and are suspected in the vae-1d|CRIMS non-convergence.

## Supporting survey (protocol context, not a width anchor)

**Blind Hyperspectral Unmixing Using Autoencoders: A Critical Comparison**
B. Palsson, J. R. Sveinsson, M. O. Ulfarsson.
*IEEE Journal of Selected Topics in Applied Earth Observations and Remote
Sensing, 2022.* Citations at retrieval: ~123.
Retrieval: <https://prod.api.sciencecopilot.modelfactory.amazon.dev/paper/redirect/bFKiaupsyz>

Useful for the paper's related-work paragraph on spectral-autoencoder design
conventions (what "a normal 1-D spectral autoencoder" looks like in practice).

---

## How the capacity confound is handled without width matching

- Params column in every results table (`utils/check-model-params.py` audit).
- Capacity points: vae-1d and vae-3d (the two baselines smaller than PRISM)
  retrained at ≈PRISM's parameter count, seed 42, IIRS + AVIRIS
  (`utils/check-model-params.py --solve-capacity`).
- Latent RATE remains matched exactly across all models
  (`utils/match_latent_rate.py --exact --check`).
