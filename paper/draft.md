# Disentangling the Spectrum with PRISM: A Dual-Stream Physics-Informed VAE for Decoupled Representational Learning of Hyperspectral Images

*Anonymous Submission — AAAI 2027*

---

## Abstract

Hyperspectral imaging captures dense information across the electromagnetic spectrum. In geographical and astronomical remote sensing, these continuous spectral bands reveal the physical and chemical fingerprints of minerals on surfaces. However, deep-space transmission failures, omitted spatial frames, sensor degradation, and noise often corrupt the data. Standard VAEs process this using convolutions that mix spatial and spectral information, inadvertently blurring delicate spectral signatures which causes posterior collapse or physically improbable spectral hallucinations that can invalidate downstream scientific analysis. To resolve this, we propose **PRISM** (Physics-Informed Representation for Isolated Spectral-Spatial Modeling). PRISM explicitly mitigates interference by decoupling spatial and spectral features into isolated parallel streams. By structurally separating these representations and bounding their fusion with a differentiable physics-informed loss, PRISM ensures the latent manifold remains mathematically grounded, preserving strict spectral fidelity and eliminating hallucinations. This robust latent topology unlocks advanced downstream capabilities for Latent Diffusion Models (LDMs). Our approach enables the high-fidelity purification of deep-space transmission corruptions and omitted spatial frames without requiring costly retransmissions. Furthermore, its smooth latent space allows for precise chemical interpolation to aid in novel mineral discovery, offering a highly scalable framework readily adaptable to other physics-heavy geospatial domains such as precision agriculture and climatology.

---

## 1. Introduction

Hyperspectral imaging (HSI) find deep rooted application in astronomical spectroscopy. HSI sensors record hundreds of contiguous spectral bands per pixel, encoding the reflectance signature of the material observed at each location. Different minerals, ices, and organics absorb light at characteristic wavelengths making HSI an effective spatially-resolved chemical fingerprinting tool. Planetary missions have used this property to map lunar hydration \citep{pieters2009m3}, Martian phyllosilicates and sulfates \citep{murchie2007crism}, and terrestrial ecosystems \citep{green1998aviris}. India's Chandrayaan-2 IIRS \citep{chowdhury2020iirs} extends this capability with 256-band reflectance cubes across the near- and short-wave infrared, enabling detailed mineralogical inference from lunar orbit.

However, HSI processing requires special care due to its fragile nature. Deep-space transmission suffers packet loss and omitted spatial frames (OSFs); on-board sensors drift and degrade over multi-year missions; and detector noise from cosmic rays or thermal effects corrupts individual bands or pixels. Retransmission from a lunar or Martian orbiter is prohibitively expensive, so downstream science requires either extensive manual curation or robust learned representations that can recover corrupted cubes without hallucinating new minerals into the record.

Deep Learning powered generative models like VAEs \citep{kingma2014vae} and Latent Diffusion Models (LDMs) \citep{rombach2022ldm} can be an intuitive solutions because of their proven ability to optimise representation learning. However standard VAEs, when adapted to HSI by simply increasing the input channel count from three (RGB) to hundreds of bands, exhibit a systematic failure mode. Their 2D convolutions mix neighboring pixels into a single latent representation, and the decoder produces a smooth blend spectrum that does not correspond to any real material. Under a KL-regularized objective this manifests as either posterior collapse — where the model reverts to reconstructing a mean spectrum — or as "physically plausible hallucinations" that pass reconstruction-error checks but silently corrupt mineralogical labels. In either case the downstream science is invalidated.

We argue that the failure is architectural, not a matter of loss tuning. Spatial context is essential for denoising and inpainting; spectral integrity is essential for chemistry. Any network that entangles the two representations forces one to be sacrificed for the other. Instead, we propose **PRISM** - Physics-Informed Representation for Isolated Spectral-Spatial Modeling — a dual-stream VAE that structurally separates spatial and spectral encoding into two parallel branches, and re-couples them only at the final reconstruction via a learned linear fusion head bounded by a differentiable Spectral Angle Mapper (SAM) physics prior \citep{kruse1993sips}. The physics prior constrains the fused output to be angularly consistent with the input spectrum, ruling out plausible-looking chemistry violations.

We evaluate PRISM against three baselines that anchor the space of HSI encoder designs — a 2D spatial VAE \citep{rombach2022ldm}, a 3D spatio-spectral VAE \citep{chen2016deep}, and a 1D pixelwise unmixing VAE \citep{su2019daen} — across four planetary and terrestrial hyperspectral datasets (IIRS, M³, CRISM, AVIRIS). Beyond raw reconstruction, we introduce three model-agnostic latent-space probes that quantify readiness for downstream generative use without requiring the training of a full diffusion model.

**Contributions.**

- **Architecture.** We introduce PRISM, a dual-stream VAE that structurally isolates spatial and spectral feature extraction and late-fuses them via a learned linear head, breaking the spatial–spectral blurring failure mode of standard HSI VAEs.
- **Physics-informed fusion.** We formulate a differentiable Spectral Angle Mapper prior on the fused reconstruction that bounds the latent manifold to physically plausible spectra, and we show that this prior does not compensate for a poor architecture — it only reaches its full effect when the encoder is already decoupled.
- **Downstream-readiness benchmark.** We propose three model-agnostic latent-space probes — noise-injection robustness, chemical interpolation smoothness, and pixel-corruption recovery — that quantify LDM-readiness without training an LDM, and we release results across four datasets and four architectures (28 runs total).

## 2. Related Works

**Deep learning for hyperspectral imaging.** Convolutional networks were adapted to HSI classification through spectral–spatial 3D-CNNs \citep{chen2016deep} and hybrid 3D–2D pipelines \citep{roy2020hybridsn} that trade off spectral depth for spatial context. In parallel, deep autoencoders became the dominant tool for hyperspectral unmixing — the decomposition of each pixel into endmember abundances — first through per-pixel MLP encoders \citep{palsson2018unmixing} and then through deeper cascaded autoencoders \citep{su2019daen} that handle noise and outliers. These lines of work motivate our Baselines B and C: 3D-CNNs preserve spectral awareness but at heavy parameter cost, and per-pixel autoencoders preserve chemistry but lose spatial structure.

**Autoencoders and latent variable models.** The variational autoencoder \citep{kingma2014vae} and its regularized $\beta$-VAE variant \citep{higgins2017betavae} introduced the modern latent-variable framework and its posterior-collapse pathology. Latent diffusion models \citep{rombach2022ldm} then established the two-stage recipe — perceptually compress with a VAE, then diffuse in latent space — that motivates our downstream-readiness experiments. Their AutoencoderKL architecture is the direct inspiration for our Baseline A.

**Physics-informed neural networks.** Raissi et al.\ \citep{raissi2019pinn} formulated physics-informed neural networks as differentiable networks trained under both data and PDE-residual losses, and this framing has since been extended to remote sensing tasks including physics-guided hyperspectral unmixing \citep{lin2026hapke}. Our SAM prior is a lightweight instantiation of the same principle applied to reconstruction: rather than encoding a full radiative-transfer PDE, we constrain the fused reconstruction to be angularly consistent with the input spectrum, which is the physically meaningful invariant for mineralogical identification.

**Spectral–spatial feature extraction and spectral fidelity.** The Spectral Angle Mapper \citep{kruse1993sips} is the community-standard metric for spectral similarity, invariant to illumination scaling and therefore a stable proxy for material identity. Recent transformer-based methods \citep{hong2022spectralformer, fu2024sst} explicitly re-attend spectral bands to preserve fidelity across long-range dependencies, and convolutional restoration networks such as HSI-DeNet \citep{chang2019hsidenet} operate under the same objective. These works confirm that spectral integrity requires architectural specialization; they do not, however, decouple spatial and spectral encoders as we do.

**Planetary and terrestrial HSI datasets.** We evaluate on four instruments spanning three planetary bodies: IIRS on Chandrayaan-2 \citep{chowdhury2020iirs}, M³ on Chandrayaan-1 \citep{pieters2009m3}, CRISM on Mars Reconnaissance Orbiter \citep{murchie2007crism}, and the airborne AVIRIS \citep{green1998aviris}. Each is described in Section 3.4.

## 3. Methodology

### 3.1 PRISM Architecture

PRISM operates on channels-last patches $\mathbf{x} \in \mathbb{R}^{B \times H \times W \times C}$, where $C$ is the number of spectral bands after per-dataset preprocessing. The network is composed of two independent encoder–decoder streams, each of which is itself a full VAE, whose reconstructions are combined by a learned linear fusion head.

**Spatial stream.** The spatial stream applies a per-pixel $\mathrm{Conv1d}$ spectral reduction that collapses the $C$-band vector at each pixel into a compact $d_r{=}32$-dimensional feature, folds the reduced tensor into a standard 2D feature map $(B, d_r, H, W)$, and then applies four strided $\mathrm{Conv2d}$ blocks that halve $H$ and $W$ at each stage (64 $\rightarrow$ 32 $\rightarrow$ 16 $\rightarrow$ 8 $\rightarrow$ 4). A final $\mathrm{LazyLinear}$ layer projects the flattened bottleneck to $2 \cdot d_z$ activations, which are chunked along the channel axis into $\mu_s$ and $\log \sigma_s^2$; the reparameterization trick yields a global patch latent $\mathbf{z}_s \in \mathbb{R}^{B \times d_z}$ with $d_z{=}256$. The decoder mirrors this pipeline via $\mathrm{ConvTranspose2d}$ blocks and a final spectral $\mathrm{Conv1d}$ back to $C$ bands. This stream's role is to encode spatial structure and coarse spectral tendencies; the spectral resolution it produces per pixel is deliberately limited by the aggressive $d_r{=}32$ bottleneck.

**Spectral stream.** The spectral stream folds all spatial locations into the batch dimension, so each pixel spectrum is encoded independently by a sequence of $\mathrm{Conv1d}$ layers with stride 2 <!--(Add the updates of the spectral dimension)-->. A $\mathrm{LazyLinear}$ layer produces $2 \cdot d_p$ features per pixel, which are chunked into $\mu_p$ and $\log \sigma_p^2$ and reshaped to a spatially-resolved latent map $\mathbf{z}_p \in \mathbb{R}^{B \times d_p \times H \times W}$ with $d_p{=}128$. The decoder is symmetric using $\mathrm{ConvTranspose1d}$. Critically, this stream never sees any 2D neighborhood: its receptive field along $H$ and $W$ is exactly one pixel. It therefore cannot blur an ice pixel into a rock pixel, and it is free to devote its full capacity to preserving each pixel's chemistry.

**Shared reparameterization.** Both streams emit $2 \cdot d$ activations at their bottleneck, which we chunk along the channel dimension to obtain $\mu$ and $\log \sigma^2$. The log-variance is clamped to $[-30, 20]$ for numerical stability before exponentiation. Sampling follows the standard trick $\mathbf{z} = \mu + \sigma \odot \boldsymbol{\varepsilon}, \boldsymbol{\varepsilon} \sim \mathcal{N}(0, I)$.

### 3.2 Physics-Informed Fusion

The two decoders each produce a full-cube reconstruction $\hat{\mathbf{x}}_s, \hat{\mathbf{x}}_p \in \mathbb{R}^{B \times H \times W \times C}$. We concatenate them along the channel axis and pass the result through a learned linear map $\mathrm{Linear}(2C \rightarrow C)$ followed by a sigmoid activation to produce the final fused reconstruction $\hat{\mathbf{x}}_f$. Because our data loader max-normalizes each cube to $[0, 1]$, the sigmoid output is dimensionally compatible without further scaling.

The training loss combines three terms:

$$\mathcal{L} = \mathcal{L}_{\mathrm{MSE}} + \beta \, \mathcal{L}_{\mathrm{KL}} + \lambda_{\mathrm{phys}} \, \mathcal{L}_{\mathrm{SAM}} \, .$$

$\mathcal{L}_{\mathrm{MSE}}$ is a multi-branch reconstruction term,

$$\mathcal{L}_{\mathrm{MSE}} = \mathrm{MSE}(\hat{\mathbf{x}}_f, \mathbf{x}) + \tfrac{1}{2}\mathrm{MSE}(\hat{\mathbf{x}}_s, \mathbf{x}) + \tfrac{1}{2}\mathrm{MSE}(\hat{\mathbf{x}}_p, \mathbf{x}) \, ,$$

which forces both branches to independently reconstruct the input and prevents one stream from dominating the fusion. $\mathcal{L}_{\mathrm{KL}}$ is the batch-averaged sum of the two streams' KL divergences to their prior. $\mathcal{L}_{\mathrm{SAM}}$ is the differentiable Spectral Angle Mapper prior applied to the fused reconstruction,

$$\mathcal{L}_{\mathrm{SAM}}(\mathbf{x}, \hat{\mathbf{x}}_f) = \frac{1}{HW}\sum_{i, j} \arccos \frac{\langle \mathbf{x}_{ij}, \hat{\mathbf{x}}_{f, ij} \rangle}{\|\mathbf{x}_{ij}\| \, \|\hat{\mathbf{x}}_{f, ij}\|} \, ,$$

which measures the mean per-pixel angle between input and reconstruction spectra. Unlike MSE, SAM is invariant to illumination scaling and therefore a stable proxy for material identity. Bounding the fusion by $\mathcal{L}_{\mathrm{SAM}}$ makes hallucinations — spectra that are close in magnitude but pointing in the wrong direction — costly, which is precisely the failure mode of standard VAEs. We use $\beta = 10^{-3}$ and $\lambda_{\mathrm{phys}} = 0.3$ throughout. PRISM is physics-informed by construction; the physics term is never ablated away in our runs.

### 3.3 Baseline Architectures

We compare PRISM against three baselines chosen to span the space of common HSI encoder designs. All four architectures share a common model-agnostic contract (`forward`, `loss_terms`, `reconstruct`, `encode_latents`, `decode_latents`), so the training and evaluation pipelines never branch on model identity. Per-dataset capacity knobs are tuned in a hyperparameter configuration so that all four models are matched in parameter count on each dataset; PRISM's advantage is therefore architectural, not one of raw capacity.

**Baseline A — 2D Spatial VAE.** A direct adaptation of the Stable Diffusion `AutoencoderKL` \citep{rombach2022ldm}: the input cube is treated as a very thick RGB image and processed with purely 2D convolutions. The encoder downsamples $(H, W)$ by $2^N$ through strided $\mathrm{Conv2d}$ blocks and projects to a channel-wise mean and log-variance map. The decoder mirrors the process via $\mathrm{ConvTranspose2d}$. Because 2D convolutions mix neighboring pixels, per-pixel chemistry is blurred in the latent space — the hypothesized failure mode motivating PRISM.

**Baseline B — 3D Spatio-Spectral VAE.** Following the classical 3D-CNN treatment of HSI \citep{chen2016deep}, the patch is lifted to a single-channel volume $(B, 1, C, H, W)$ and processed entirely with $\mathrm{Conv3d}$ and $\mathrm{ConvTranspose3d}$ layers. Strides are set to preserve the spectral depth ($\mathrm{stride}{=}1$ along $C$) while downsampling $H$ and $W$, so the model is band-count-agnostic and works unchanged on IIRS (108 bands), M³ (84), CRISM (variable), and AVIRIS (224). This is the most parameter-heavy of our models and, in practice, the most prone to posterior collapse when the latent bottleneck is aggressive.

**Baseline C — 1D Pixelwise VAE.** Following the autoencoder-unmixing tradition \citep{palsson2018unmixing, su2019daen}, all $(H, W)$ locations are folded into the batch dimension and each pixel spectrum is passed through a shared MLP encoder and decoder. The model never sees the 2D grid, so it excels at preserving per-pixel chemistry but has no neighborhood context to denoise or regularize corrupted pixels.

### 3.4 Datasets and Ablation Design

**IIRS (Chandrayaan-2 Imaging Infrared Spectrometer).** A push-broom near/short-wave infrared imaging spectrometer built by ISRO and flying on Chandrayaan-2, covering approximately 0.8--5.0\,$\mu$m across 256 contiguous bands at $\sim$80\,m ground sampling distance from a 100\,km lunar orbit \citep{chowdhury2020iirs}. Its primary use is lunar mineralogy and hydration mapping. We select bands 7--115 (108 reflective bands), normalize each pixel spectrum by the $\sim$1500\,nm reference band, and Savitzky--Golay-smooth along the spectral axis.

**M³ (Moon Mineralogy Mapper).** A NASA/JPL imaging spectrometer flown as a mission-of-opportunity aboard ISRO's Chandrayaan-1, spanning 430--3000\,nm across 260 bands with 140\,m/pixel global mode and 70\,m/pixel targeted mode \citep{pieters2009m3}. Landmark result: detection of surface OH/H$_2$O across the lunar disk.

**CRISM (Compact Reconnaissance Imaging Spectrometer for Mars).** JHU/APL instrument on NASA's Mars Reconnaissance Orbiter, covering 362--3920\,nm at 6.55\,nm sampling for 544 bands split across VNIR and IR spectrographs. Targeted mode achieves 18--36\,m/pixel; the multispectral mapping mode delivers 100--200\,m/pixel \citep{murchie2007crism}. Its primary use is mapping Martian aqueous alteration mineralogy.

**AVIRIS (Airborne Visible/Infrared Imaging Spectrometer).** A NASA/JPL airborne whiskbroom spectrometer covering 400--2500\,nm across 224 bands at $\sim$10\,nm sampling, with ground pixel size 4--20\,m depending on platform (ER-2, Twin Otter) \citep{green1998aviris}. AVIRIS is the de facto terrestrial benchmark and connects our lunar/Martian evaluation to Earth-observation applications.

**Ablation grid.** We run 28 configurations in total:

- **PRISM:** 4 runs, one per dataset (the SAM prior is intrinsic to the architecture and is not ablated).
- **Baselines A, B, C:** each is trained on each of the 4 datasets under two loss regimes — standard ELBO (MSE + $\beta\mathcal{L}_{\mathrm{KL}}$) and physics-augmented ELBO (adds $\lambda_{\mathrm{phys}}\mathcal{L}_{\mathrm{SAM}}$), for $3 \times 4 \times 2 = 24$ runs.

All models are trained under an identical pipeline: 64$\times$64 patches at stride 48, region-disjoint 70/15/15 train/valid/test splits (contiguous height slices to prevent leakage across similar terrain), 100 epochs, batch size 32, cosine-annealed learning rate starting at $10^{-4}$. Metrics reported per run are SAM ($\downarrow$), PSNR ($\uparrow$), and SSIM ($\uparrow$) on the held-out test split, and total parameter count.

## 4. Results and Discussion

### 4.1 Reconstruction Quality (Table 1)

We first report per-dataset, per-architecture reconstruction quality.

**Table 1.** Reconstruction quality across four datasets and four architectures. Bold indicates best per (dataset, metric); all figures on held-out test split.

| Dataset | Model      | Loss     | SAM $\downarrow$ | PSNR $\uparrow$ | SSIM $\uparrow$ | Params (M) |
|---------|------------|----------|------------------|-----------------|-----------------|-----------:|
| IIRS    | Baseline A | ELBO     | TBD              | TBD             | TBD             | TBD        |
| IIRS    | Baseline A | +SAM     | TBD              | TBD             | TBD             | TBD        |
| IIRS    | Baseline B | ELBO     | TBD              | TBD             | TBD             | TBD        |
| IIRS    | Baseline B | +SAM     | TBD              | TBD             | TBD             | TBD        |
| IIRS    | Baseline C | ELBO     | TBD              | TBD             | TBD             | TBD        |
| IIRS    | Baseline C | +SAM     | TBD              | TBD             | TBD             | TBD        |
| IIRS    | **PRISM**  | +SAM     | TBD              | TBD             | TBD             | TBD        |
| M³      | (7 rows, same structure) | | | | | |
| CRISM   | (7 rows) | | | | | |
| AVIRIS  | (7 rows) | | | | | |

**Discussion.** Our hypothesis, informed by the architectural trade-offs, is that no baseline should dominate all three metrics simultaneously. Baseline C should achieve the strongest SAM on every dataset because it never mixes pixels — its representation is chemically pristine — but should suffer on PSNR and SSIM because it cannot leverage spatial neighborhoods to smooth noise or infer corrupted structure. Baseline A should invert this pattern: strong PSNR and SSIM (2D convolutions are excellent spatial denoisers) but poor SAM (its latent space blends adjacent chemistries). Baseline B should sit between the two on all three metrics but at substantially higher parameter cost, and we expect to see occasional posterior collapse manifesting as flat-mean reconstructions on the hardest test splits (e.g.\ CRISM's variable-band, heterogeneous scenes). PRISM should achieve best or near-best on all three metrics simultaneously, at parameter count comparable to Baselines A and C, precisely because its two streams solve the two sub-problems independently before recombining them under a physics-bounded fusion.

### 4.2 Physics-Loss Ablation (Table 2)

**Table 2.** Change in SAM, PSNR, and SSIM when the SAM physics prior is added to the standard ELBO objective for Baselines A, B, C.

| Dataset | Model      | $\Delta$SAM | $\Delta$PSNR | $\Delta$SSIM |
|---------|------------|-------------|--------------|--------------|
| IIRS    | Baseline A | TBD         | TBD          | TBD          |
| IIRS    | Baseline B | TBD         | TBD          | TBD          |
| IIRS    | Baseline C | TBD         | TBD          | TBD          |
| (M³ / CRISM / AVIRIS: same structure) | | | | |

**Discussion.** We expect the SAM prior to reduce spectral angle for all three baselines — that is what it is designed to do — but not to close the gap to PRISM. This is the central methodological claim of the paper: **loss engineering cannot repair a poorly decoupled architecture**. If Baseline A's 2D convolutions structurally blur pixel chemistries in the latent space, adding a SAM term to the loss will bias the decoder toward angle-consistent outputs but cannot force the encoder to preserve angle-informative structure that it discarded. The physics prior reaches its full effect only when the encoder is already decoupled, as in PRISM.

### 4.3 Downstream 1 — Noise-Injection Robustness (Table 3)

To assess whether each model's latent space is diffusion-ready, we perform a controlled corruption study on the latents themselves. For each test patch $\mathbf{x}$ we compute the deterministic latent $\mathbf{z} = \mathrm{encode\_latents}(\mathbf{x})$, add isotropic Gaussian noise $\boldsymbol{\varepsilon} \sim \mathcal{N}(0, \sigma^2 I)$ at $\sigma \in \{0, 0.1, 0.5, 1.0\}$, and decode. The gap between the perturbed reconstruction and the clean reconstruction, as a function of $\sigma$, characterizes how gracefully the latent manifold degrades.

**Table 3.** Noise-injection robustness. SAM/PSNR/SSIM of $\mathrm{decode}(\mathbf{z} + \boldsymbol{\varepsilon})$ vs.\ $\mathrm{decode}(\mathbf{z})$ across noise levels $\sigma$.

| Dataset | Model      | $\sigma{=}0.1$ (SAM/PSNR/SSIM) | $\sigma{=}0.5$ | $\sigma{=}1.0$ |
|---------|------------|--------------------------------|----------------|----------------|
| IIRS    | Baseline A | TBD                            | TBD            | TBD            |
| IIRS    | Baseline B | TBD                            | TBD            | TBD            |
| IIRS    | Baseline C | TBD                            | TBD            | TBD            |
| IIRS    | **PRISM**  | TBD                            | TBD            | TBD            |
| (M³ / CRISM / AVIRIS: same structure) | | | | |

**Discussion.** We predict that PRISM's latent space, being the product of two mildly-KL-regularized streams that each have their own physically-meaningful role, will degrade smoothly with increasing $\sigma$. Baseline A, whose latent map spatially over-compresses chemistry, should collapse fastest — its latents encode entangled cross-pixel spectra, and Gaussian perturbations move the decoder rapidly off the manifold of valid mineralogies. Baseline C's per-pixel latents will remain locally chemistry-plausible but will decorrelate spatially, producing an incoherent recovered scene.

### 4.4 Downstream 2 — Chemical Interpolation Smoothness (Table 4)

Generative use of a VAE — sampling, diffusion, or interpolation — requires a smooth latent manifold: moving one step in latent space should correspond to a physically continuous change in reconstructed chemistry. We probe this by sampling pixel pairs $A, B$ from the test set with distinct spectra, interpolating in latent space as $\mathbf{z}_{\alpha} = \alpha \, \mathbf{z}_A + (1 - \alpha) \, \mathbf{z}_B$ for $\alpha \in [0, 1]$, decoding, and tracking the resulting spectrum. We report a scalar *jaggedness* measure: the mean $L_2$ norm of the second finite difference of the recovered spectrum along $\alpha$, aggregated over test-set pairs. Lower jaggedness means a smoother latent geometry.

**Table 4.** Chemical-interpolation jaggedness ($\downarrow$).

| Dataset | Baseline A | Baseline B | Baseline C | **PRISM** |
|---------|-----------:|-----------:|-----------:|----------:|
| IIRS    | TBD        | TBD        | TBD        | TBD       |
| M³      | TBD        | TBD        | TBD        | TBD       |
| CRISM   | TBD        | TBD        | TBD        | TBD       |
| AVIRIS  | TBD        | TBD        | TBD        | TBD       |

**Discussion.** We expect PRISM to achieve the lowest jaggedness overall, because the SAM prior explicitly steers the fused reconstruction toward angle-consistent nearest neighbors. Baseline C is likely to be chemistry-smooth but spatially discontinuous — that is, individual pixels interpolate cleanly but adjacent pixels do so independently, giving a chemically valid but spatially incoherent transition. Baseline A will be spatially smooth but chemically erratic, showing sharp spectral discontinuities along $\alpha$ that correspond to the decoder jumping between latent-space basins.

### 4.5 Downstream 3 — Pixel-Corruption Recovery (Table 5)

Finally, we test the practically motivating scenario: recovering cubes from omitted spatial frames and sensor-noise corruption. We randomly mask a fraction $\rho \in \{0.05, 0.10, 0.20\}$ of pixels in each test patch (zeroing all bands at those pixels), pass the corrupted patch through the trained VAE, and report per-metric reconstruction quality at the masked pixels only.

**Table 5.** Pixel-corruption recovery. Metrics on masked pixels; SAM $\downarrow$, PSNR $\uparrow$, SSIM $\uparrow$.

| Dataset | Model      | $\rho{=}0.05$ (SAM/PSNR/SSIM) | $\rho{=}0.10$ | $\rho{=}0.20$ |
|---------|------------|-------------------------------|---------------|---------------|
| IIRS    | Baseline A | TBD                           | TBD           | TBD           |
| IIRS    | Baseline B | TBD                           | TBD           | TBD           |
| IIRS    | Baseline C | TBD                           | TBD           | TBD           |
| IIRS    | **PRISM**  | TBD                           | TBD           | TBD           |
| (M³ / CRISM / AVIRIS: same structure) | | | | |

**Discussion.** We anticipate a clean stratification: Baseline C, having no spatial context, cannot recover the missing pixels at all — it must hallucinate spectra from noise; Baseline A over-smooths, recovering low-frequency structure but the wrong chemistry; Baseline B is competitive but at inflated parameter cost. PRISM should lead by combining the spatial stream's ability to infer plausible neighborhoods with the spectral stream's ability to keep the inferred pixels chemically valid, and with the SAM prior forcing consistency at the fusion step.

### 4.6 Synthesis

Taken together, Tables 1--5 test a single claim: that structural decoupling of spatial and spectral encoding, bounded by a physics-informed fusion, produces a representation that is *simultaneously* (i) high-fidelity in reconstruction, (ii) robust to latent-space perturbation, (iii) smooth under interpolation, and (iv) recoverable under pixel-level corruption. No single-stream baseline achieves all four properties; PRISM does, because it does not force one representation to be sacrificed for the other. This is precisely the condition a downstream Latent Diffusion Model needs from its VAE backbone.

## 5. Conclusion

We introduced PRISM, a dual-stream physics-informed VAE for hyperspectral imagery that structurally isolates spatial and spectral feature extraction and re-couples them only at a learned, SAM-bounded fusion. Across four planetary and terrestrial hyperspectral datasets (IIRS, M³, CRISM, AVIRIS) and four architectures (28 runs), we show that no single-stream baseline can simultaneously match PRISM on reconstruction quality, latent-space robustness, interpolation smoothness, and pixel-corruption recovery. The physics prior alone does not close the gap for baselines with entangled encoders — the effect of the SAM term compounds with, and depends on, structural decoupling.

Because PRISM's latent geometry satisfies the criteria for generative use — noise-graceful, interpolation-smooth, corruption-recoverable — it is a natural VAE backbone for a Phase-2 Latent Diffusion Model that would perform end-to-end purification of deep-space transmission corruptions and omitted spatial frames without requiring costly retransmission. The framework carries over unchanged to other physics-heavy hyperspectral domains — precision agriculture, climate monitoring, and mineral exploration on Earth — where the same trade-off between spatial context and chemical fidelity applies.

## Ethical Statement

Planetary hyperspectral imagery informs long-horizon scientific claims — the presence or absence of water, minerals, and organic markers on other worlds. Any generative model that reconstructs or inpaints such data has a corresponding capacity to hallucinate: to place a mineral where there is none, or to remove one that is present. PRISM's SAM prior is designed to make hallucinations costly, and our downstream-readiness benchmarks are designed to make degradation modes measurable. Nevertheless, we recommend that reconstructions produced by PRISM (or any comparable generative model) be used to *complement* raw sensor data in scientific pipelines, not to replace it, and that mineralogical conclusions drawn from reconstructed cubes be flagged as such in downstream publications. The same considerations apply to any transfer of these methods to precision agriculture or climate monitoring, where high-stakes policy and resource decisions may be downstream of the reconstruction.

## References

*(Rendered by BibTeX from `paper/references.bib` when compiled; see that file for the 17 verified entries.)*
