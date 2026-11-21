# PCTF fusion-ablation grid — results and interpretation (run of 2026-11-18)

> **Run:** `scripts/retrain.sh` at `hpc@76f17dc` ("add primal-dual constrained optimizer, explicit uncertainty tolerance, and band-shared gate"). It ran on the lab box (1× NVIDIA RTX PRO 4000 Blackwell, bf16 autocast) from **18.11.2026 10:02 to 19.11.2026 13:14**, about 27 h.
> **Sources:** Telegram export `ChatExport_2026-11-20/` (`messages.html` plus the final cumulative CSVs `ablation_table (3).csv`, `downstream_table (3).csv`, `probes (3).csv`, `stats (3).csv`) and a canonical extraction of those files. Corrections to that extraction are listed in [Appendix A](#appendix-a-data-provenance-and-corrections).
> **Conventions:** every "mean ± x" is the mean over seeds 67 and 69 ± the **half-range** $\tfrac12\lvert v_{67}-v_{69}\rvert$. With two seeds this is the only honest spread; it is not a standard deviation or a confidence interval. Angles are converted from radians with $\theta[^\circ]=\theta[\text{rad}]\cdot 180/\pi$. A number marked *derived* was computed here from logged quantities, and its formula is given next to it.

---

## 1. How this experiment works

### 1.1 The question

PRISM, the model in the paper, is a two-stream VAE. A **spatial stream** reconstructs the patch as $\hat x_s$ and a **spectral stream** reconstructs it as $\hat x_p$. The two are blended into the final output:

$$
\hat x_f \;=\; \alpha\odot\hat x_s + (1-\alpha)\odot\hat x_p,\qquad \alpha\in(0,1).
$$

The paper trains this with a **SAM penalty**, a fixed weight $\lambda=0.3$ on the mean spectral angle. The PCTF grid ("Physics-Constrained Trust-Region Fusion") asks two questions:

1. **Does the way the streams are fused matter?** The options are a fixed 50/50 blend, a learned gate without physics, and a learned gate with physics.
2. **Does it matter whether spectral fidelity is a *penalty* or a *constraint*?** A penalty adds $\lambda\cdot\text{SAM}$ with a fixed $\lambda$. A constraint requires $\mathbb E[\text{SAM}]\le\varepsilon$ and lets an optimiser find $\lambda$. Can the constraint be made per-pixel and uncertainty-aware?

Five variants answer these. All five share the same encoder/decoder backbone and differ **only in the fusion module and the loss**.

### 1.2 Shared backbone (identical in all five variants)

The code is `modules/vae_fusion_pctf.py:HSI_DualStream_Fusion_Base`. It subclasses `HSI_DualStream_PI_VAE` (`modules/vae_our.py`) and replaces two components:

* **Spatial stream = `SpatialEncoderDecoderNL`** (`modules/vae_our_variants.py`). This is the spatial stream of registry model **`vae-our-nl`**, the architecture the paper calls PRISM.
  * The encoder starts with a 3×3 conv from $C$ bands to 64 channels. Three stride-2 3×3 conv blocks follow (64→128→256→512 channels, 64×64 → 8×8), and then a 1×1 conv produces $(\mu_s,\log\sigma^2_s)$. The latent is $z_s\in\mathbb R^{d_s\times 8\times 8}$.
  * The decoder mirrors the encoder with transposed convs, ends in a 3×3 conv back to $C$ bands, and applies a sigmoid.
* **Spectral stream = `SpectralEncoderDecoder`** (unchanged). This is a per-pixel 1D conv encoder and decoder, so pixels are never mixed. Its latent is $z_p\in\mathbb R^{d_p\times 64\times 64}$.
* **Reparameterisation:** $z=\mu+\sigma\odot\epsilon$ with $\epsilon\sim\mathcal N(0,I)$ and $\log\sigma^2$ clamped to $[-30,20]$. The code samples $z$ **also at evaluation time**, so every reconstruction metric comes from one posterior sample.
* **Latent rate:** this is matched to the common budget $T=64\,d_s+4096\,d_p$ from the per-dataset YAMLs:

| dataset | bands $C$ | $d_s$ | $d_p$ | latent elements | probe `rate_dev_pct` |
|---|---|---|---|---|---|
| IIRS | 256 | 64 | 3 | 16,384 | 0.0 % |
| AVIRIS | 424 | 128 | 5 | 28,672 | −5.4 % |
| M3 | 84 | 4 | 1 | 4,352 | **+23.5 %** (inside the ±25 % P2 tolerance, but far looser than the others) |
| CRIMS | 456 | 128 | 5 | 28,672 | +1.8 % |

**Shared reconstruction and KL terms.** These are the same as the paper's Eq. (rec), with $w=$ `vae_our_aux_mse_weight` $=0.1$, which gives a 5:1:1 mix:

$$
\mathcal L_{\text{rec}}=\frac{0.5\,\mathrm{MSE}(\hat x_f,x)+0.1\,\mathrm{MSE}(\hat x_s,x)+0.1\,\mathrm{MSE}(\hat x_p,x)}{0.7},\qquad
\mathcal L_{\text{KL}}=\tfrac12\big(\mathrm{KL}(z_s)+\mathrm{KL}(z_p)\big),\qquad
\mathcal L_{\text{ELBO}}=\mathcal L_{\text{rec}}+\beta\,\mathcal L_{\text{KL}},\ \beta=10^{-3}.
$$

**Spectral angle.** For true and reconstructed spectra $x_i,\hat x_i\in\mathbb R^C$ at pixel $i$:

$$
\theta_i=\arccos\frac{\langle x_i,\hat x_i\rangle}{\lVert x_i\rVert\,\lVert\hat x_i\rVert},\qquad \text{SAM}=\frac1N\sum_i\theta_i \quad(\text{radians; }\texttt{modules/losses.py}).
$$

A pixel whose true spectrum is (numerically) all zeros contributes **exactly $\pi/2$** whatever the model outputs. **SAM-valid** is the same mean taken only over pixels with spectral energy above $10^{-8}$; it is the reported metric. Training and validation use the *raw* SAM, so they include those $\pi/2$ pixels. This matters a great deal for CRIMS (§4.2).

### 1.3 The gate actually used, and how it differs from the paper's PRISM gate

Variants 2–5 use the same gate network $g$:

$$
\alpha=\sigma\!\Big(\underbrace{\mathrm{Conv}_{1\times1}^{64\to1}\circ\mathrm{ReLU}\circ\mathrm{Conv}_{1\times1}^{2C\to64}}_{g}\big([\hat x_s;\hat x_p]\big)\Big)\in(0,1)^{B\times H\times W\times 1}.
$$

This is **one $\alpha$ per pixel, shared across all bands** (band-shared). Each pixel's gate sees only that pixel's two reconstructed spectra, because a 1×1 kernel has no spatial neighbourhood.

| | paper's PRISM gate (`AdaptiveGatedFusion`, used by `vae-our-nl`) | PCTF gate (this grid) |
|---|---|---|
| layers | Conv **3×3** (2C→64) → ReLU → Conv **3×3** (64→**C**) → sigmoid | Conv **1×1** (2C→64) → ReLU → Conv **1×1** (64→**1**) → sigmoid |
| $\alpha$ shape | per pixel **and per band**, $(B,H,W,C)$ | per pixel, **shared across bands**, $(B,H,W,1)$ |
| receptive field | 3×3 neighbourhood (5×5 through two layers) | the pixel only |

The band-shared design is intentional (module docstring). A per-band $\alpha$ can splice a spectrum from bands of different streams and so distort its shape. A band-shared $\alpha$ makes $\hat x_{f,i}$ a convex combination of two whole spectra. **Consequence:** variant 3 is *not* the paper's PRISM. It is "PRISM's backbone and loss with a different gate". The `vae-our-nl` reference rows (§3) are the paper's architecture. Their difference from variant 3 measures the gate change, together with any code drift between runs (§4.7).

### 1.4 The five variants

| # | registry name | `--loss` | fusion | training objective |
|---|---|---|---|---|
| 1 | `vae-fusion-fixed` | physics | $\alpha\equiv 0.5$ (no parameters) | $\mathcal L_{\text{ELBO}}+0.3\,\text{SAM}$ |
| 2 | `vae-fusion-unconstrained` | **standard** | learned gate $g$ | $\mathcal L_{\text{ELBO}}$ only |
| 3 | `vae-fusion-sam-penalty` | physics | learned gate $g$ | $\mathcal L_{\text{ELBO}}+0.3\,\text{SAM}$ |
| 4 | `vae-fusion-constrained` | physics | learned gate $g$ + inference-time trust region | primal–dual: $\min\mathcal L_{\text{ELBO}}$ s.t. $\text{SAM}\le\varepsilon=0.08$ rad |
| 5 | `vae-fusion-uncertainty` | physics | learned gate $g$ + per-pixel trust region | primal–dual on per-pixel violations $\big(\theta_i-\varepsilon_i\big)_+$ |

In every variant SAM is computed on the **fused** output, $\text{SAM}(x,\hat x_f)$.

#### Exp 1: fixed weight

$$\hat x_f=0.5\,\hat x_s+0.5\,\hat x_p,\qquad \mathcal L=\mathcal L_{\text{ELBO}}+\lambda\,\text{SAM},\ \lambda=0.3.$$

This is the "no learned fusion" control.

#### Exp 2: learned gate, no SAM ("standard" loss)

$$\mathcal L=\mathcal L_{\text{ELBO}}=\mathcal L_{\text{rec}}+\beta\,\mathcal L_{\text{KL}}.$$

**"Standard" means exactly this: the 5:1:1 reconstruction MSE plus $10^{-3}\times$ the KL term, with no SAM term.** In the code, `loss_mode="unconstrained"` ignores the `use_physics` flag altogether. The `--loss standard` argument therefore only changes the checkpoint file stem (`vae-fusion-unconstrained_standard_seed*.pt`) and how the row is labelled. SAM is still *computed and logged*, but it never enters the gradient, so the SAM of Exp 2 is a by-product of an MSE trajectory.

#### Exp 3: learned gate + SAM penalty

$$\mathcal L=\mathcal L_{\text{ELBO}}+\lambda\,\text{SAM},\quad\lambda=0.3\ \text{(fixed)}.$$

This is the paper's objective (Eq. objective) with the band-shared 1×1 gate.

#### Exp 4: primal–dual constrained fusion (PCTF)

The intended problem is

$$\min_{\theta}\ \mathcal L_{\text{ELBO}}(\theta)\quad\text{s.t.}\quad \mathbb E[\text{SAM}(x,\hat x_f)]\le\varepsilon,\qquad \varepsilon=0.08\ \text{rad}=4.58^\circ,$$

and its Lagrangian is $J(\theta,\lambda)=\mathcal L_{\text{ELBO}}+\lambda(\text{SAM}-\varepsilon)$ with $\lambda\ge0$. The code implements it as follows:

* The multiplier is $\lambda=\mathrm{softplus}(\ell)$, where $\ell$ is an `nn.Parameter` initialised at $-1.04$, so $\lambda_0=\ln(1+e^{-1.04})\approx0.303$.
* The loss passed to the optimiser is
  $$\mathcal L=\mathcal L_{\text{ELBO}}+\underbrace{\operatorname{sg}(\lambda)\,\text{SAM}}_{\text{primal step}}\;\underbrace{-\;\lambda\,\big(\operatorname{sg}(\text{SAM})-\varepsilon\big)}_{\text{dual step}},$$
  where $\operatorname{sg}$ is stop-gradient (`.detach()`). The network weights see $\lambda\cdot\nabla\text{SAM}$. The multiplier sees $\partial\mathcal L/\partial\ell=-\sigma(\ell)\,(\text{SAM}-\varepsilon)$, so **minimising** $\mathcal L$ increases $\lambda$ when $\text{SAM}>\varepsilon$ and decreases it when $\text{SAM}<\varepsilon$. That is dual ascent.
* The dual step uses the **same AdamW optimiser** as the network: lr $10^{-3}$, 5-epoch warmup then cosine decay, weight decay $10^{-5}$, gradient-norm clip 1.0. Adam normalises each parameter's step, so $\ell$ moves by roughly one learning-rate-sized step per iteration *whatever the size of the violation*. Its speed is set by the LR schedule rather than by how badly the constraint is violated. §4.1 checks this against the logs.
* **The logged loss is not a Lagrangian you can compare.** Numerically the two $\lambda$ terms cancel except for $\lambda\varepsilon$:
  $$\mathcal L\;=\;\mathcal L_{\text{ELBO}}+\lambda\,\text{SAM}-\lambda\,\text{SAM}+\lambda\varepsilon\;=\;\mathcal L_{\text{ELBO}}+\lambda\,\varepsilon .$$
  So `val_loss` for Exp 4 grows with $\lambda$ even when nothing else changes. This identity lets us back the multiplier out of the logs as an *implied* $\lambda=(\text{val\_loss}-\text{val\_mse}-\beta\,\text{val\_kld})/\varepsilon$ (§4.1).
* **Inference-time trust region.** The gate outputs a candidate $\tilde\alpha$. When `self.training` is False, the model measures the disagreement between its own streams, $d_i=\theta(\hat x_{s,i},\hat x_{p,i})$, which needs no ground truth, and contracts the spatial weight:
  $$\alpha_i=\tilde\alpha_i\cdot\min\!\Big(1,\ \frac{\varepsilon}{d_i+10^{-7}}\Big).$$
  Where the streams disagree by more than $\varepsilon$, fusion moves toward the spectral stream. **This contraction is active during validation** (`model.eval()`). Val SAM is what selects the best-SAM checkpoint and drives early stopping, and it also covers every test and downstream number. The gradients, however, always see the uncontracted $\tilde\alpha$. This **train/eval mismatch** is a confound (§4.3).

#### Exp 5: uncertainty-aware constrained fusion (U-PCTF)

Exp 5 uses a per-pixel tolerance $\varepsilon_i\in[\varepsilon_{\min},\varepsilon_{\max}]=[0.03,0.12]$ rad $=[1.72^\circ,6.88^\circ]$, computed without ground truth from three normalised cues:

$$
\tilde d_i=\frac{d_i}{\overline{d}},\quad d_i=\theta(\hat x_{s,i},\hat x_{p,i});\qquad
\tilde s_i=\frac{s_i}{\overline{s}},\quad s_i=\frac{\operatorname{mean}_c\hat x_{p,i,c}}{\operatorname{std}_c\hat x_{p,i,c}};\qquad
\tilde u_i=\frac{u_i}{\overline{u}},\quad u=\operatorname{up}_{8\to64}\!\Big(\operatorname{mean}_{\text{ch}}\,e^{\frac12\log\sigma^2_s}\Big),
$$

where each bar is the per-patch spatial mean. These combine into

$$
\tau_i=\sigma\big(\tilde s_i-1.5\,\tilde d_i-\tilde u_i\big),\qquad \varepsilon_i=\varepsilon_{\min}+(\varepsilon_{\max}-\varepsilon_{\min})\,\tau_i .
$$

High stream disagreement or high spatial-latent uncertainty tightens the tolerance, and a high spectral SNR relaxes it. With per-pixel violation $v_i=(\theta(x_i,\hat x_{f,i})-\varepsilon_i)_+$, the loss is

$$
\mathcal L=\mathcal L_{\text{ELBO}}+\operatorname{sg}(\lambda)\,\overline{v^2}\;-\;\lambda\,\operatorname{sg}(\bar v),\qquad \lambda=\mathrm{softplus}(\ell),\ \ell_0=-1.04 .
$$

The inference contraction is $\alpha_i=\tilde\alpha_i\min(1,\varepsilon_i/d_i)$. Four properties of this implementation matter for interpretation:

1. **The dual target is 0 and $v_i\ge0$**, so $\partial\mathcal L/\partial\ell=-\sigma(\ell)\,\bar v\le0$. In practice $\lambda$ **only grows**.
2. **The primal penalty is a squared hinge on angles in radians.** For $v_i<1$ rad (every valid pixel), $v_i^2<v_i$, so the pressure is weak. On CRIMS the $\pi/2$ zero-energy pixels give $v_i\approx1.45$–$1.54>1$ and **dominate** the penalty (§4.2).
3. **$\varepsilon_i$ is not detached.** The network can reduce the violation by *raising its own tolerance*, either by making the streams agree, raising $\tilde s$, or shrinking $\sigma_s$, instead of fitting $x$ better.
4. **Downstream and probes use a different $\varepsilon_i$ from the reconstruction table.** `decode_latents` (used by downstream.py and the P3 swap probe) passes `logvar_s=None`, so there $\tilde u_i\equiv1$. `reconstruct` (used by inference.py) uses the real $\log\sigma^2_s$.

The logged loss is $\mathcal L_{\text{ELBO}}+\lambda(\overline{v^2}-\bar v)$. It is **negative** whenever violations are below 1 rad and $\lambda$ is large, which is why Exp 5 shows negative `val_loss` on IIRS, AVIRIS and M3.

### 1.5 Pipeline (`scripts/retrain.sh`)

**Grid:** 4 datasets (IIRS, AVIRIS, M3, CRIMS) × 5 variants × 2 seeds (67, 69) = **40 cells**. They ran in this order: per dataset, Exp 1 → Exp 5, with seeds innermost. Each cell runs four steps in sequence:

1. **Train**: `train/train.py --model <m> --dataset <ds> --loss <l> --seed <s> --epochs 60 --patience 3`. It emits Telegram `[START]`, `[HB]` every 10 epochs, and `[OK]`/`[STOP]`/`[FAIL]` at the end.
2. **Infer**: `inference/inference.py --select sam`. Test-split reconstruction metrics over the **whole** test split (IIRS 3,084 patches; AVIRIS 1,950; M3 3,959; CRIMS 369).
3. **Probes**: `inference/probes.py`. P2 latent rate, P3 collapse (active units, latent swap), P4 spatial-shuffle SRI, run on a 512-patch sample of the test split (all 369 on CRIMS). P1/P5 columns are NaN in this run.
4. **Downstream**: `inference/downstream.py`. Latent noise injection and latent interpolation (§1.6).

A cell is marked done (`results/.done/<DS>__<stem>_seed<N>`) only if steps 2–4 all succeed. After each dataset, `verdict.py` and `aggregate.py` post the cumulative CSVs to Telegram. Settings below were confirmed from the 39 `[START]` messages in the export (the fortieth was not exported, see Appendix A):

| setting | value | where it comes from |
|---|---|---|
| epochs | 60 | CLI (= YAML) |
| early-stopping patience | **3** (M3's YAML says 10; the CLI overrides it; `[START]` confirms 3 for M3) | CLI |
| early-stopping rule | stop when **neither** val SAM nor val `mse_final` has improved for 3 consecutive epochs | `train.py` |
| checkpoint evaluated | `_bestsam.pt` = epoch of minimum **val raw SAM** (`--select sam`) | `retrain.sh` |
| optimiser | AdamW, lr $10^{-3}$, wd $10^{-5}$, grad-clip 1.0; LR warmup 0.01×→1× over 5 epochs, then cosine over the remaining 55 | YAML / `train.py` |
| $\beta$, $\lambda$ | $10^{-3}$, 0.3 (Exp 1, 3; Exp 4/5 start at $\lambda_0\approx0.303$) | YAML |
| batch size | IIRS 32, M3 32, AVIRIS 16, CRIMS 16 | YAML; confirmed by `[START]` |
| precision | bf16 autocast | `[START]` |

### 1.6 What each output file and column means

**`ablation_table.csv`** has one row per (dataset, model, loss, seed, select) and comes from `inference.py` on the full test split.

| column | meaning |
|---|---|
| `mse` | pooled MSE of the **fused** reconstruction vs. input (data in [0, 1]) |
| `psnr` | $10\log_{10}(1/\text{mse})$, dB |
| `ssim` | 11×11 Gaussian-windowed, per-band SSIM (`modules/metrics.py`) |
| `sam_rad` | raw mean spectral angle over all pixels (rad), **including** $\pi/2$ zero-energy pixels |
| `sam_valid`, `valid_pixel_frac` | mean angle over pixels with energy $>10^{-8}$ (rad), and the fraction of such pixels |
| `sid`, `sid_clamped_frac` | spectral information divergence (symmetric KL between band-normalised spectra, bands clamped at $10^{-8}$); `sid_clamped_frac` = fraction of **input** pixels with at least one band $\le10^{-8}$. This is a property of the data and identical for every model: IIRS 0.979, AVIRIS 1.000, M3 0.061, CRIMS 0.608. **On IIRS, AVIRIS and CRIMS, SID is strongly affected by the clamp; it is not ranked here.** |
| `scc` | spatial correlation coefficient (correlation of Laplacian high-pass images) |
| `q2n` | Q2ⁿ hypercomplex quality index, block size 32 |

**`downstream_table.csv`** has one row per (dataset, model, loss, seed) and comes from `downstream.py` with its defaults.

* The input is **one fixed batch of 8 test patches**: the first batch of the unshuffled test loader. The noise RNG seed is 42.
* **Noise injection:** encode to the deterministic latents $\mu$ of both streams, add $\mathcal N(0,\sigma^2)$ noise with **absolute** $\sigma\in\{0,0.1,0.5,1.0\}$ to *both* latents, decode, and score against the clean input. The table keeps two levels:
  * `psnr_clean` = PSNR at $\sigma=0$;
  * **`psnr_mid` = PSNR at $\sigma=0.5$, the single "middle" noise level kept**;
  * `psnr_drop` = `psnr_clean` − `psnr_mid`;
  * `sam_mid` = **raw** SAM (rad) at $\sigma=0.5$.
* **Interpolation:** $z(\alpha)=\alpha z_A+(1-\alpha)z_B$ between patches 0 and 1 at 11 values of $\alpha$, tracking the decoded spectrum at pixel (32, 32). Two quantities are reported:
  * `jaggedness` = mean $\ell_2$ norm of the second difference along $\alpha$;
  * `path_length` = summed $\ell_2$ step length.
* **jag / path**, in the canonical summary, is the ratio of the seed means.
* This is a very low-power measurement: 8 patches, one noise draw, one interpolation path and one pixel.

**`probes.csv`** has one row per cell and comes from `probes.py` on the 512-patch sample.

| column | meaning |
|---|---|
| `epochs` | epoch of the evaluated (best-SAM) checkpoint |
| `latent_elements`, `rate_dev_pct`, `rate_matched` | P2, the latent budget |
| `active_units` | P3, fraction of latent dimensions whose aggregate posterior KL exceeds 0.01 nats |
| `latent_swap_delta` | P3, relative change in SAM when every patch is decoded from a different patch's latent. `collapsed=True` if this is below 0.02 or fewer than 1 % of units are active |
| `sri` | P4, relative SAM change after shuffling pixel positions |

**`stats.csv`** is from `verdict.py` → `inference/stats.py`. It holds paired comparisons **of one seed against the same seed**, per dataset and metric (psnr, ssim, sam, sam_valid), computed **on the probe sample** (512 patches; 369 on CRIMS, and 260 for CRIMS `sam_valid`, the patches with valid pixels). It gives paired bootstrap CIs, a permutation p-value, Holm correction (`p_holm`) and Cliff's delta. `meaningful` means above the preregistered effect floor (0.5 dB, 0.005 rad, 0.01 SSIM), and `verdict` ∈ {`<cell>_better`, `significant_but_negligible`, `no_difference`}. Cells flagged collapsed by P3 are excluded, and cells without probes are absent. Its deltas are per-patch means on that subsample, so **they do not equal the differences in the full-test-split table**.

### 1.7 Reading the Telegram messages

| tag | when | contents |
|---|---|---|
| `[START]` | cell begins | resolved hyper-parameters, GPU, run id `#nnnnn` |
| `[HB]` | every 10 epochs | train/val `loss`, `mse`, `sam`, `kld`; val `psnr`, `ssim`; best val SAM so far |
| `[OK]` | finished all 60 epochs | epochs, wall time, peak GPU, `best @ epoch N: <best val SAM in rad>`, a table every 10 epochs plus the last epoch |
| `[STOP]` | early stopping fired before 60 epochs | same as `[OK]`; this is normal termination, **not a failure** |
| `[FAIL]` | training crashed | traceback (none occurred) |
| CSV posts | after each dataset | cumulative `ablation_table`, `downstream_table`, `probes`, `stats` (the "(3)" files are the final ones) |
| `🏁 Retrain Grid Finished` | end | `Completed` = cells whose four steps all succeeded; `Failed` = number of failed *steps*, not cells |

Inside `[HB]`/`[OK]`/`[STOP]`:

* `val_mse` is the **training reconstruction term $\mathcal L_{\text{rec}}$** (the 5:1:1 mix), not the fused-output MSE.
* `val_loss` is the full objective, so for Exp 4/5 it includes the $\lambda$ terms (§1.4).
* `val_sam` is **raw** SAM in rad.
* PSNR and SSIM are computed on the validation split's fused reconstruction.

---

## 2. Run accounting

| | count | evidence |
|---|---|---|
| cells planned | 40 | launch message, `retrain.sh` |
| cells trained (`[OK]` or `[STOP]`) | **40** | 3 `[OK]` + 37 `[STOP]`, 0 `[FAIL]` |
| finished all 60 epochs (`[OK]`) | **3** | IIRS Exp 1 s67 and s69; IIRS Exp 3 s69 |
| early-stopped (`[STOP]`) | **37** | stopped between 7 and 51 epochs (§3.1) |
| inference rows (select = sam) | 40 / 40 | `ablation_table (3).csv` |
| downstream rows | 40 / 40 | `downstream_table (3).csv` |
| probes rows | **34 / 40** | `probes (3).csv` |
| final summary | `Completed: 34/40 · Skipped: 0 · Failed: 6` | last message |

**"Failed: 6", resolved.** All six failures are the **probes step on AVIRIS** for Exp 3, Exp 4 and Exp 5, both seeds. These are exactly the six cells missing from `probes (3).csv`. Training, inference and downstream succeeded for all six, so their reconstruction and downstream numbers below are valid. Their stats rows are also missing, because stats are built from probe outputs. **The root cause is not in the export**: the probe failure did not post a traceback to Telegram. On the lab box, `run_stage` appended the last 20 lines of each failed stage to `logs/retrain_pctf_20261118-100225.log`, so the cause should be recorded there.

Two observations constrain the cause. Neither proves it.

* The six failures are the **last six AVIRIS cells in run order**. Exp 3 uses the *same* fusion-module class (a 1×1 gate) as Exp 2, whose AVIRIS probes succeeded, and every M3 and CRIMS cell probed fine afterwards. A cause specific to the model code is therefore less likely than one tied to the environment or to state at that point in the run, but this is untested.
* There is a precedent: a **2026-09-17 AVIRIS probes OOM**. AVIRIS at 512 patches × 424 bands is 3.3 GiB per full reconstruction tensor. It is documented in the `inference/probes.py` comments, was addressed by chunked metric accumulation in commit `6f1d3bd` ("fix: OOM on AVIRIS"), and was not reproduced here for Exp 1/2. Whether this is the same failure is unknown.

**These six cells have no `.done` marker.** A plain re-run of `retrain.sh` would **retrain them and overwrite their checkpoints**. Re-run `inference/probes.py` for them directly instead (§5).

**M3** was run here (10 cells) but **M3 is not in the paper**. Its results are reported for completeness. M3 cells also show KL instability (§4.6).

---

## 3. Results

How to read the tables below:

* **Bold rows** are the five PCTF variants, retrained in this run.
* *Italic rows* are **reference rows carried over from the earlier clean-slate grid** (seeds 67/69, 60 epochs; patience 3 per CLAUDE.md). They were re-exported in the same CSVs and **not retrained here**. The code commit and any protocol drift for those runs cannot be verified from this export.
  * **M3 exception:** commit `7c9030c` raised the M3 YAML patience from 3 to 10, and `1bff419` scheduled an M3 retrain under patience 10. The M3 reference rows may therefore come from patience-10 runs. This is unverified, and it further confounds every M3 comparison against them.
* `vae-our-nl` is the **same architecture** as the paper's PRISM. Its numbers here are *not* the paper's Table values: the paper reports IIRS 6.70° / 40.22 dB, this export 7.011° / 39.97 dB. The paper's "CRISM" is the dataset `CRIMS`.
* NA means the column was not computed for those carried-over runs.
* † marks a carried-over metric present for **seed 69 only**, so there is no spread.
* **ckpt epoch** is the epoch of the evaluated best-SAM checkpoint. For the reference rows it comes from `probes.csv` ("?" = no probe row).

### 3.1 Training length per cell (undertraining is visible here)

From the `[OK]`/`[STOP]` messages. "best val SAM" is raw SAM on the validation split, in rad, and for Exp 4/5 it already includes the inference-time contraction.

| Dataset | Variant | status s67 / s69 | epochs run s67 / s69 | best-SAM epoch s67 / s69 | best val SAM (rad) s67 / s69 | final val PSNR (dB) s67 / s69 | wall s67 / s69 |
|---|---|---|---|---|---|---|---|
| IIRS | Exp 1 fixed α=0.5 | OK / OK | 60 / 60 | 58 / 59 | 0.1213 / 0.1222 | 37.15 / 37.45 | 1h11m13s / 1h11m04s |
| IIRS | Exp 2 learned gate, no SAM | STOP / STOP | 16 / 14 | 13 / 11 | 0.2814 / 0.2833 | 35.66 / 34.16 | 18m33s / 16m16s |
| IIRS | Exp 3 learned gate + λ·SAM | **STOP / OK** | **18 / 60** | 15 / 60 | 0.1338 / 0.1201 | 32.61 / 38.91 | 21m00s / 1h09m41s |
| IIRS | Exp 4 primal-dual (PCTF) | STOP / STOP | 14 / 23 | **7 / 8** | 0.1832 / 0.1838 | 28.68 / 30.23 | 16m20s / 26m44s |
| IIRS | Exp 5 uncertainty (U-PCTF) | STOP / STOP | 18 / 19 | 14 / 14 | 0.1986 / 0.2207 | 30.88 / 30.95 | 21m39s / 22m50s |
| AVIRIS | Exp 1 fixed α=0.5 | STOP / STOP | 18 / 30 | 15 / 26 | 0.0298 / 0.0286 | 34.53 / 34.67 | 33m58s / 51m07s |
| AVIRIS | Exp 2 learned gate, no SAM | STOP / STOP | 26 / 44 | 23 / 41 | 0.0448 / 0.0440 | 33.34 / 33.76 | 43m26s / 1h13m26s |
| AVIRIS | Exp 3 learned gate + λ·SAM | STOP / STOP | 33 / 15 | 30 / 11 | 0.0284 / 0.0300 | 36.09 / 35.88 | 1h40m12s / 52m36s |
| AVIRIS | Exp 4 primal-dual (PCTF) | STOP / STOP | **12 / 12** | 9 / 9 | 0.0329 / 0.0331 | 34.63 / 34.67 | 41m31s / 44m18s |
| AVIRIS | Exp 5 uncertainty (U-PCTF) | STOP / STOP | 51 / 35 | 48 / 32 | 0.0353 / 0.0357 | 34.32 / 34.11 | 3h01m37s / 2h06m58s |
| M3 | Exp 1 fixed α=0.5 | STOP / STOP | **7** / 28 | **4** / 25 | 0.0498 / 0.0408 | 31.47 / 32.60 | 6m19s / 23m23s |
| M3 | Exp 2 learned gate, no SAM | STOP / STOP | **7 / 10** | **4 / 5** | 0.0757 / 0.0793 | 28.42 / 28.17 | 6m03s / 8m37s |
| M3 | Exp 3 learned gate + λ·SAM | STOP / STOP | 19 / 23 | 16 / 20 | 0.0419 / 0.0416 | 31.13 / 32.28 | 14m40s / 14m56s |
| M3 | Exp 4 primal-dual (PCTF) | STOP / STOP | 16 / 37 | 13 / 28 | 0.0495 / 0.0508 | 29.56 / 32.88 | 6m44s / 15m30s |
| M3 | Exp 5 uncertainty (U-PCTF) | STOP / STOP | 20 / 13 | 17 / 10 | 0.0603 / 0.0628 | 28.03 / 28.56 | 8m39s / 5m40s |
| CRIMS | Exp 1 fixed α=0.5 | STOP / STOP | 25 / 29 | 21 / 26 | 0.4843 / 0.4833 | 34.39 / 33.80 | 16m56s / 17m43s |
| CRIMS | Exp 2 learned gate, no SAM | STOP / STOP | 17 / 20 | 14 / 17 | 0.5060 / 0.5058 | 32.74 / 33.19 | 10m28s / 12m17s |
| CRIMS | Exp 3 learned gate + λ·SAM | STOP / STOP | 44 / 23 | 41 / 20 | 0.4823 / 0.4837 | 37.17 / 36.36 | 27m09s / 14m18s |
| CRIMS | Exp 4 primal-dual (PCTF) | STOP / STOP | 42 / 29 | 39 / 26 | 0.4808 / 0.4817 | 31.42 / 31.78 | 25m53s / 17m55s |
| CRIMS | Exp 5 uncertainty (U-PCTF) | STOP / STOP | 18 / 12 | 15 / 9 | 0.4971 / 0.4988 | 23.76 / 25.74 | 11m35s / 12m05s |

The LR schedule is a 5-epoch warmup followed by a cosine over 55 epochs. A cell that stopped at epoch 7–20 was still at or near **full learning rate** and never reached the low-LR phase. IIRS Exp 1 shows what that phase is worth: its val PSNR jumped **32.62 → 36.51 dB between epochs 30 and 40** (seed 67 heartbeats).

### 3.2 IIRS (256 bands; test n = 3,084)

| Model | SAM-valid (°) ↓ | PSNR (dB) ↑ | SSIM ↑ | SID ↓ | SCC ↑ | Q2ⁿ ↑ | MSE (×10⁻⁴) ↓ | ckpt epoch (s67/s69) |
|---|---|---|---|---|---|---|---|---|
| **Exp 1 fixed α=0.5** | 7.224 ± 0.032 | 35.91 ± 0.18 | 0.891 ± 0.002 | 0.1044 ± 0.0006 | 0.221 ± 0.005 | 0.756 ± 0.002 | 2.57 ± 0.10 | 58/59 |
| **Exp 2 learned gate, no SAM** ⚠ P3-collapsed | 16.522 ± 0.085 | 36.02 ± 0.01 | 0.911 ± 0.006 | 0.3348 ± 0.0013 | 0.005 ± 0.001 | 0.037 ± 0.001 | 2.50 ± 0.00 | 13/11 |
| **Exp 3 learned gate + λ·SAM** | 7.592 ± 0.435 | 34.43 ± 3.26 | 0.849 ± 0.060 | 0.1115 ± 0.0076 | 0.184 ± 0.008 | 0.682 ± 0.090 | 4.67 ± 2.97 | 15/60 |
| **Exp 4 primal-dual (PCTF)** | 11.148 ± 0.022 | 27.70 ± 0.23 | 0.615 ± 0.020 | 0.1485 ± 0.0015 | 0.177 ± 0.005 | 0.451 ± 0.018 | 17.01 ± 0.90 | 7/8 |
| **Exp 5 uncertainty (U-PCTF)** | 12.279 ± 0.579 | 29.88 ± 0.04 | 0.700 ± 0.003 | 0.1730 ± 0.0088 | 0.165 ± 0.001 | 0.521 ± 0.003 | 10.29 ± 0.10 | 14/14 |
| *ref: vae-our-nl (PRISM arch.)* | 7.011 ± 0.063 | 39.97 ± 0.32 | 0.932 ± 0.004 | NA | NA | NA | 1.01 ± 0.07 | 59/59 |
| *ref: vae-our (older PRISM, 1D proj.)* | 7.378 ± 0.003 | 31.51 ± 0.42 | 0.811 ± 0.002 | 0.1016 ± 0.0008 | 0.154 ± 0.066 | 0.634 ± 0.016 | 7.09 ± 0.69 | 60/59 |
| *ref: 2D Spatial VAE, SAM* | 9.848 ± 0.117 | 39.93 ± 0.09 | 0.946 ± 0.000 | 0.1653 ± 0.0021 | 0.074 ± 0.001 | 0.708 ± 0.009 | 1.02 ± 0.02 | 60/60 |
| *ref: 1D Pixelwise VAE, SAM* | 6.932 ± 0.001 | 32.47 ± 0.03 | 0.825 ± 0.001 | 0.0948 ± 0.0001 | 0.222 ± 0.000 | 0.639 ± 0.001 | 5.67 ± 0.04 | 51/60 |
| *ref: 3D Spatio-Spectral VAE, SAM* | 11.526 ± 0.007 | 39.19 ± 0.01 | 0.955 ± 0.000 | 0.1973 ± 0.0006 | 0.105 ± 0.000 | 0.569 ± 0.002 | 1.20 ± 0.00 | 59/60 |
| *ref: 2D Spatial VAE, std* | 16.982 ± 0.014 | 35.74 ± 0.03 | 0.911 ± 0.003 | 0.3482 † | 0.003 † | 0.042 † | 2.67 ± 0.02 | 7/8 |
| *ref: 1D Pixelwise VAE, std* | 16.493 ± 0.020 | 34.18 ± 0.03 | 0.803 ± 0.000 | 0.3344 † | 0.065 † | 0.213 † | 3.82 ± 0.02 | 22/24 |
| *ref: 3D Spatio-Spectral VAE, std* | 17.380 ± 0.016 | 36.45 ± 0.00 | 0.910 ± 0.000 | 0.3874 † | 0.001 † | 0.091 † | 2.27 ± 0.00 | 34/27 |

Per seed, IIRS Exp 3 is s67 **31.17 dB / 8.026°** (18 epochs) and s69 **37.70 dB / 7.157°** (60 epochs). The mean hides two very different models.

| Model (IIRS downstream, 8 patches) | PSNR clean (dB) | PSNR @σ=0.5 (dB) | PSNR drop (dB) | raw SAM @σ=0.5 (°) | jaggedness | path length | jag / path |
|---|---|---|---|---|---|---|---|
| **Exp 1 fixed α=0.5** | 34.15 ± 0.36 | 29.26 ± 0.35 | 4.90 ± 0.70 | 16.49 ± 0.56 | 0.0024 ± 0.0001 | 0.320 ± 0.002 | 0.0077 |
| **Exp 2 learned gate, no SAM** ⚠ | 34.49 ± 0.14 | 34.25 ± 0.09 | 0.24 ± 0.05 | 11.94 ± 0.13 | 0.0018 ± 0.0001 | 0.572 ± 0.013 | 0.0031 |
| **Exp 3 learned gate + λ·SAM** | 31.53 ± 2.89 | 28.56 ± 1.94 | 2.97 ± 0.95 | 15.94 ± 2.03 | 0.0026 ± 0.0016 | 0.257 ± 0.116 | 0.0100 |
| **Exp 4 primal-dual (PCTF)** | 30.12 ± 0.11 | 24.82 ± 0.54 | 5.30 ± 0.43 | 20.47 ± 0.29 | 0.0016 ± 0.0000 | 0.280 ± 0.009 | 0.0058 |
| **Exp 5 uncertainty (U-PCTF)** | 29.83 ± 0.24 | 24.59 ± 0.82 | 5.24 ± 0.57 | 22.51 ± 0.33 | 0.0024 ± 0.0004 | 0.216 ± 0.003 | 0.0109 |
| *ref: vae-our-nl (PRISM arch.)* | 39.23 ± 0.85 | 30.67 ± 0.03 | 8.57 ± 0.88 | 14.49 ± 0.11 | 0.0029 ± 0.0002 | 0.403 ± 0.013 | 0.0072 |
| *ref: vae-our* | 29.70 ± 0.42 | 26.37 ± 0.15 | 3.32 ± 0.56 | 21.10 ± 1.07 | 0.0017 ± 0.0003 | 0.143 ± 0.039 | 0.0119 |
| *ref: 2D Spatial VAE, SAM* | 38.98 ± 0.09 | 36.59 ± 0.10 | 2.40 ± 0.01 | 7.91 ± 0.20 | 0.0095 ± 0.0002 | 0.488 ± 0.023 | 0.0196 |
| *ref: 1D Pixelwise VAE, SAM* | 30.59 ± 0.04 | 29.26 ± 0.02 | 1.32 ± 0.06 | 9.77 ± 0.26 | 0.0243 ± 0.0012 | 0.381 ± 0.006 | 0.0639 |
| *ref: 3D Spatio-Spectral VAE, SAM* | 38.07 ± 0.03 | 36.34 ± 0.00 | 1.73 ± 0.03 | 10.06 ± 0.02 | 0.0053 ± 0.0007 | 0.548 ± 0.003 | 0.0096 |

### 3.3 AVIRIS (424 bands; test n = 1,950)

| Model | SAM-valid (°) ↓ | PSNR (dB) ↑ | SSIM ↑ | SID ↓ | SCC ↑ | Q2ⁿ ↑ | MSE (×10⁻⁴) ↓ | ckpt epoch (s67/s69) |
|---|---|---|---|---|---|---|---|---|
| **Exp 1 fixed α=0.5** | 1.630 ± 0.028 | 34.25 ± 0.00 | 0.869 ± 0.000 | 0.0015 ± 0.0001 | 0.522 ± 0.004 | 0.858 ± 0.000 | 3.76 ± 0.00 | 15/26 |
| **Exp 2 learned gate, no SAM** | 2.412 ± 0.018 | 33.62 ± 0.09 | 0.830 ± 0.003 | 0.0033 ± 0.0001 | 0.276 ± 0.008 | 0.658 ± 0.002 | 4.34 ± 0.09 | 23/41 |
| **Exp 3 learned gate + λ·SAM** (no probes) | 1.633 ± 0.035 | 35.81 ± 0.22 | 0.870 ± 0.001 | 0.0021 ± 0.0002 | 0.529 ± 0.009 | 0.872 ± 0.005 | 2.63 ± 0.13 | 30/11 |
| **Exp 4 primal-dual (PCTF)** (no probes) | 1.822 ± 0.002 | 34.69 ± 0.03 | 0.842 ± 0.003 | 0.0031 ± 0.0002 | 0.438 ± 0.005 | 0.790 ± 0.008 | 3.40 ± 0.02 | 9/9 |
| **Exp 5 uncertainty (U-PCTF)** (no probes) | 1.988 ± 0.008 | 33.87 ± 0.17 | 0.824 ± 0.001 | 0.0022 ± 0.0000 | 0.370 ± 0.006 | 0.717 ± 0.002 | 4.10 ± 0.16 | 48/32 |
| *ref: vae-our-nl (PRISM arch.)* | 1.632 ± 0.001 | 35.83 ± 0.12 | 0.872 ± 0.002 | NA | NA | NA | 2.61 ± 0.07 | 20/17 |
| *ref: vae-our (older PRISM, 1D proj.)* | 1.631 ± 0.015 | 35.74 ± 0.15 | 0.868 ± 0.002 | 0.0018 ± 0.0002 | 0.516 ± 0.009 | 0.868 ± 0.003 | 2.67 ± 0.09 | 25/22 |
| *ref: 2D Spatial VAE, SAM* | 5.428 ± 0.232 | 14.78 ± 0.07 | 0.720 ± 0.002 | 0.0224 ± 0.0010 | 0.000 ± 0.000 | 0.124 ± 0.063 | 333.01 ± 5.53 | 4/8 |
| *ref: 1D Pixelwise VAE, SAM* | 1.619 ± 0.048 | 34.93 ± 0.72 | 0.849 ± 0.006 | 0.0017 ± 0.0000 | 0.460 ± 0.012 | 0.808 ± 0.010 | 3.26 ± 0.53 | 17/60 |
| *ref: 3D Spatio-Spectral VAE, SAM* | 1.973 ± 0.186 | 33.24 ± 1.39 | 0.839 ± 0.017 | 0.0029 ± 0.0007 | 0.089 ± 0.083 | 0.676 ± 0.122 | 4.99 ± 1.55 | 28/5 |
| *ref: 2D Spatial VAE, std* | 3.555 ± 0.310 | 29.01 ± 0.45 | 0.814 ± 0.003 | 0.0081 † | 0.007 † | 0.366 † | 12.62 ± 1.31 | ?/11 |
| *ref: 1D Pixelwise VAE, std* | 2.695 ± 0.239 | 32.25 ± 0.94 | 0.794 ± 0.011 | 0.0117 † | 0.262 † | 0.619 † | 6.10 ± 1.30 | ?/8 |
| *ref: 3D Spatio-Spectral VAE, std* | 2.466 ± 0.034 | 32.79 ± 0.09 | 0.824 ± 0.001 | 0.0084 † | 0.041 † | 0.583 † | 5.26 ± 0.11 | ?/30 |

| Model (AVIRIS downstream, 8 patches) | PSNR clean (dB) | PSNR @σ=0.5 (dB) | PSNR drop (dB) | raw SAM @σ=0.5 (°) | jaggedness | path length | jag / path |
|---|---|---|---|---|---|---|---|
| **Exp 1 fixed α=0.5** | 35.13 ± 0.01 | 25.89 ± 0.20 | 9.24 ± 0.21 | 2.87 ± 0.20 | 0.0062 ± 0.0004 | 1.398 ± 0.026 | 0.0045 |
| **Exp 2 learned gate, no SAM** | 33.59 ± 0.13 | 24.83 ± 0.49 | 8.76 ± 0.36 | 2.41 ± 0.16 | 0.0133 ± 0.0027 | 1.419 ± 0.005 | 0.0094 |
| **Exp 3 learned gate + λ·SAM** | 35.32 ± 0.21 | 21.18 ± 0.77 | 14.14 ± 0.55 | 4.42 ± 0.73 | 0.0070 ± 0.0006 | 1.334 ± 0.009 | 0.0053 |
| **Exp 4 primal-dual (PCTF)** | 34.92 ± 0.08 | 20.86 ± 0.29 | 14.06 ± 0.21 | 4.82 ± 0.40 | 0.0076 ± 0.0008 | 1.371 ± 0.006 | 0.0056 |
| **Exp 5 uncertainty (U-PCTF)** | 34.61 ± 0.08 | 23.56 ± 0.27 | 11.04 ± 0.19 | 2.48 ± 0.00 | 0.0117 ± 0.0018 | 1.317 ± 0.017 | 0.0089 |
| *ref: vae-our-nl (PRISM arch.)* | 35.19 ± 0.02 | 20.84 ± 0.24 | 14.35 ± 0.22 | 4.83 ± 0.17 | 0.0084 ± 0.0003 | 1.334 ± 0.034 | 0.0063 |
| *ref: vae-our* | 35.42 ± 0.21 | 20.93 ± 0.55 | 14.48 ± 0.76 | 4.66 ± 0.31 | 0.0083 ± 0.0000 | 1.347 ± 0.018 | 0.0062 |
| *ref: 2D Spatial VAE, SAM* | 14.98 ± 0.46 | 15.76 ± 0.02 | −0.78 ± 0.48 | 4.61 ± 0.28 | 0.0002 ± 0.0002 | 0.053 ± 0.053 | 0.0033 |
| *ref: 1D Pixelwise VAE, SAM* | 34.80 ± 0.82 | 26.57 ± 1.92 | 8.23 ± 1.10 | 2.20 ± 0.11 | 0.0088 ± 0.0004 | 1.264 ± 0.187 | 0.0070 |
| *ref: 3D Spatio-Spectral VAE, SAM* | 35.06 ± 0.72 | 26.54 ± 1.35 | 8.52 ± 2.07 | 4.19 ± 0.62 | 0.0047 ± 0.0006 | 1.540 ± 0.049 | 0.0031 |

### 3.4 M3 (84 bands; test n = 3,959; not in the paper)

| Model | SAM-valid (°) ↓ | PSNR (dB) ↑ | SSIM ↑ | SID ↓ | SCC ↑ | Q2ⁿ ↑ | MSE (×10⁻⁴) ↓ | ckpt epoch (s67/s69) |
|---|---|---|---|---|---|---|---|---|
| **Exp 1 fixed α=0.5** | 6.306 ± 0.444 | 30.07 ± 1.03 | 0.762 ± 0.020 | 0.1968 ± 0.0066 | 0.220 ± 0.040 | 0.369 ± 0.043 | 10.13 ± 2.36 | 4/25 |
| **Exp 2 learned gate, no SAM** | 8.662 ± 0.183 | 29.07 ± 0.28 | 0.778 ± 0.037 | 0.2113 ± 0.0032 | 0.157 ± 0.013 | 0.191 ± 0.004 | 12.42 ± 0.79 | 4/5 |
| **Exp 3 learned gate + λ·SAM** | 5.899 ± 0.050 | 30.69 ± 0.74 | 0.787 ± 0.033 | 0.1889 ± 0.0029 | 0.160 ± 0.025 | 0.422 ± 0.014 | 8.65 ± 1.46 | 16/20 |
| **Exp 4 primal-dual (PCTF)** | 6.830 ± 0.082 | 28.72 ± 1.26 | 0.733 ± 0.038 | 0.1956 ± 0.0040 | 0.177 ± 0.027 | 0.367 ± 0.033 | 14.00 ± 3.95 | 13/28 |
| **Exp 5 uncertainty (U-PCTF)** | 7.384 ± 0.053 | 26.79 ± 0.09 | 0.653 ± 0.008 | 0.1941 ± 0.0023 | 0.126 ± 0.001 | 0.236 ± 0.005 | 20.95 ± 0.43 | 17/10 |
| *ref: vae-our-nl (PRISM arch.)* | 6.298 ± 0.312 | 29.70 ± 1.69 | 0.749 ± 0.040 | NA | NA | NA | 11.53 ± 4.27 | 5/19 |
| *ref: vae-our (older PRISM, 1D proj.)* | 6.110 ± 0.486 | 30.80 ± 1.37 | 0.784 ± 0.045 | NA | NA | NA | 8.73 ± 2.66 | 9/25 |
| *ref: 2D Spatial VAE, SAM* | 5.673 ± 0.244 | 32.54 ± 0.98 | 0.857 ± 0.026 | NA | NA | NA | 5.72 ± 1.27 | 14/59 |
| *ref: 1D Pixelwise VAE, SAM* | 6.327 ± 0.047 | 27.99 ± 0.37 | 0.726 ± 0.003 | NA | NA | NA | 15.94 ± 1.37 | 17/16 |
| *ref: 3D Spatio-Spectral VAE, SAM* | 5.534 ± 0.025 | 34.22 ± 0.00 | 0.928 ± 0.001 | NA | NA | NA | 3.79 ± 0.00 | 22/32 |
| *ref: 2D Spatial VAE, std* | 8.324 ± 0.432 | 21.62 ± 8.78 | 0.794 ± 0.096 | NA | NA | NA | 264.80 ± 255.68 | 9/4 |
| *ref: 1D Pixelwise VAE, std* | 8.359 ± 0.022 | 28.77 ± 0.16 | 0.745 ± 0.015 | NA | NA | NA | 13.28 ± 0.49 | 5/7 |
| *ref: 3D Spatio-Spectral VAE, std* | 9.352 ± 1.063 | 31.91 ± 0.19 | 0.869 ± 0.012 | NA | NA | NA | 6.44 ± 0.29 | 6/9 |

| Model (M3 downstream, 8 patches) | PSNR clean (dB) | PSNR @σ=0.5 (dB) | PSNR drop (dB) | raw SAM @σ=0.5 (°) | jaggedness | path length | jag / path |
|---|---|---|---|---|---|---|---|
| **Exp 1 fixed α=0.5** | 35.96 ± 0.00 | 24.68 ± 0.06 | 11.28 ± 0.06 | 1.78 ± 0.14 | 0.0021 ± 0.0009 | 0.175 ± 0.010 | 0.0123 |
| **Exp 2 learned gate, no SAM** | 32.54 ± 0.72 | 25.12 ± 0.30 | 7.42 ± 1.03 | 2.02 ± 0.03 | 0.0007 ± 0.0002 | 0.044 ± 0.004 | 0.0164 |
| **Exp 3 learned gate + λ·SAM** | 35.63 ± 0.98 | 23.42 ± 0.43 | 12.20 ± 1.40 | 2.18 ± 0.12 | 0.0019 ± 0.0002 | 0.205 ± 0.021 | 0.0093 |
| **Exp 4 primal-dual (PCTF)** | 36.20 ± 1.20 | 24.02 ± 0.56 | 12.19 ± 0.64 | 2.44 ± 0.75 | 0.0020 ± 0.0005 | 0.173 ± 0.021 | 0.0114 |
| **Exp 5 uncertainty (U-PCTF)** | 33.91 ± 0.01 | 24.25 ± 0.68 | 9.66 ± 0.70 | 1.92 ± 0.07 | 0.0005 ± 0.0000 | 0.052 ± 0.013 | 0.0088 |
| *ref: vae-our-nl (PRISM arch.)* | 34.73 ± 1.88 | 22.79 ± 0.48 | 11.94 ± 2.36 | 2.81 ± 0.23 | 0.0022 ± 0.0003 | 0.191 ± 0.006 | 0.0114 |
| *ref: vae-our* | 36.22 ± 1.76 | 26.53 ± 3.49 | 9.69 ± 1.73 | 1.91 ± 0.83 | 0.0020 ± 0.0001 | 0.186 ± 0.006 | 0.0110 |
| *ref: 2D Spatial VAE, SAM* | 26.66 ± 3.46 | 27.64 ± 1.75 | −0.98 ± 5.21 | 1.07 ± 0.12 | 0.0042 ± 0.0004 | 0.200 ± 0.006 | 0.0211 |
| *ref: 1D Pixelwise VAE, SAM* | 33.08 ± 0.08 | 24.75 ± 0.11 | 8.33 ± 0.03 | 1.47 ± 0.02 | 0.0009 ± 0.0001 | 0.240 ± 0.002 | 0.0039 |
| *ref: 3D Spatio-Spectral VAE, SAM* | 42.11 ± 0.30 | 24.05 ± 0.04 | 18.06 ± 0.26 | 5.35 ± 0.02 | 0.0014 ± 0.0002 | 0.268 ± 0.005 | 0.0053 |

### 3.5 CRIMS (456 bands; test n = 369; valid-pixel fraction 0.498)

| Model | SAM-valid (°) ↓ | PSNR (dB) ↑ | SSIM ↑ | SID ↓ | SCC ↑ | Q2ⁿ ↑ | MSE (×10⁻⁴) ↓ | ckpt epoch (s67/s69) |
|---|---|---|---|---|---|---|---|---|
| **Exp 1 fixed α=0.5** | 2.850 ± 0.034 | 30.38 ± 0.11 | 0.741 ± 0.020 | 0.2266 ± 0.0088 | 0.417 ± 0.000 | 0.388 ± 0.003 | 9.16 ± 0.23 | 21/26 |
| **Exp 2 learned gate, no SAM** | 4.491 ± 0.101 | 32.66 ± 0.17 | 0.897 ± 0.017 | 0.1911 ± 0.0064 | 0.368 ± 0.004 | 0.316 ± 0.003 | 5.42 ± 0.21 | 14/17 |
| **Exp 3 learned gate + λ·SAM** | 2.760 ± 0.062 | 33.96 ± 0.28 | 0.872 ± 0.013 | 0.2607 ± 0.0185 | 0.444 ± 0.004 | 0.424 ± 0.003 | 4.03 ± 0.26 | 41/20 |
| **Exp 4 primal-dual (PCTF)** | **2.609 ± 0.063** | 29.13 ± 0.47 | 0.594 ± 0.016 | 0.2072 ± 0.0121 | 0.505 ± 0.003 | 0.449 ± 0.004 | 12.28 ± 1.33 | 39/26 |
| **Exp 5 uncertainty (U-PCTF)** | 3.966 ± 0.078 | 25.92 ± 2.19 | 0.466 ± 0.024 | 0.0826 ± 0.0061 | 0.376 ± 0.005 | 0.318 ± 0.003 | 28.92 ± 13.45 | 15/9 |
| *ref: vae-our-nl (PRISM arch.)* | 2.655 ± 0.047 | 33.95 ± 0.82 | 0.814 ± 0.046 | NA | NA | NA | 4.10 ± 0.76 | 27/60 |
| *ref: vae-our (older PRISM, 1D proj.)* | 2.684 ± 0.095 | 33.73 ± 0.54 | 0.800 ± 0.017 | 0.2208 ± 0.0330 | 0.453 ± 0.002 | 0.426 ± 0.007 | 4.27 ± 0.53 | 23/59 |
| *ref: 2D Spatial VAE, SAM* | 6.115 ± 0.077 | 23.77 ± 0.08 | 0.801 ± 0.006 | 0.3769 ± 0.0194 | 0.028 ± 0.010 | 0.213 ± 0.006 | 41.95 ± 0.78 | 4/9 |
| *ref: 1D Pixelwise VAE, SAM* | 3.711 ± 0.492 | 29.46 ± 1.42 | 0.760 ± 0.062 | 0.1976 ± 0.0024 | 0.394 ± 0.016 | 0.339 ± 0.026 | 11.93 ± 3.77 | 8/15 |
| *ref: 3D Spatio-Spectral VAE, SAM* | 2.823 ± 0.043 | 27.13 ± 0.29 | 0.927 ± 0.013 | 0.0672 ± 0.0047 | 0.046 ± 0.004 | 0.362 ± 0.008 | 19.39 ± 1.28 | 13/17 |
| *ref: 2D Spatial VAE, std* | 5.953 ± 0.436 | 25.59 ± 0.41 | 0.923 ± 0.010 | 0.7456 † | 0.047 † | 0.235 † | 27.74 ± 2.62 | 35/19 |
| *ref: 1D Pixelwise VAE, std* | 4.911 ± 0.054 | 30.59 ± 0.40 | 0.896 ± 0.009 | 0.1205 † | 0.390 † | 0.315 † | 8.77 ± 0.80 | 19/25 |
| *ref: 3D Spatio-Spectral VAE, std* | 4.070 ± 0.284 | 27.26 ± 0.56 | 0.932 ± 0.010 | 2.8676 † | 0.031 † | 0.290 † | 18.96 ± 2.44 | 26/8 |

Raw SAM (all pixels) on CRIMS is 46.1–47.9° for **every** model, and the five fusion variants span only 46.10–47.12°. That figure measures the share of zero-energy pixels, not model quality (§4.2). Use SAM-valid.

| Model (CRIMS downstream, 8 patches) | PSNR clean (dB) | PSNR @σ=0.5 (dB) | PSNR drop (dB) | raw SAM @σ=0.5 (°) | jaggedness | path length | jag / path |
|---|---|---|---|---|---|---|---|
| **Exp 1 fixed α=0.5** | 32.06 ± 0.93 | 23.58 ± 0.14 | 8.49 ± 1.07 | 18.91 ± 0.20 | 0.4443 ± 0.0213 | 14.600 ± 0.072 | 0.0304 |
| **Exp 2 learned gate, no SAM** | 32.98 ± 0.06 | 23.64 ± 0.37 | 9.34 ± 0.42 | 18.85 ± 0.30 | 0.5736 ± 0.0477 | 15.421 ± 0.064 | 0.0372 |
| **Exp 3 learned gate + λ·SAM** | 35.95 ± 0.09 | 18.46 ± 0.06 | 17.48 ± 0.15 | 20.96 ± 0.15 | 0.4518 ± 0.0259 | 14.955 ± 0.083 | 0.0302 |
| **Exp 4 primal-dual (PCTF)** | 31.61 ± 0.52 | 18.84 ± 0.19 | 12.76 ± 0.33 | 21.40 ± 0.19 | 0.3749 ± 0.0208 | 14.083 ± 0.220 | 0.0266 |
| **Exp 5 uncertainty (U-PCTF)** | 27.28 ± 0.80 | 20.34 ± 0.29 | 6.94 ± 0.50 | 21.02 ± 0.05 | 0.3843 ± 0.0309 | 13.617 ± 0.299 | 0.0282 |
| *ref: vae-our-nl (PRISM arch.)* | 35.44 ± 0.23 | 18.49 ± 0.92 | 16.94 ± 1.15 | 20.85 ± 0.04 | 0.5093 ± 0.0282 | 15.143 ± 0.111 | 0.0336 |
| *ref: vae-our* | 36.57 ± 0.32 | 19.72 ± 0.36 | 16.85 ± 0.67 | 20.54 ± 0.04 | 0.5290 ± 0.0001 | 15.107 ± 0.024 | 0.0350 |
| *ref: 2D Spatial VAE, SAM* | 22.27 ± 0.07 | 23.83 ± 0.29 | −1.56 ± 0.37 | 19.55 ± 0.22 | 0.5697 ± 0.0365 | 15.134 ± 0.174 | 0.0376 |
| *ref: 1D Pixelwise VAE, SAM* | 26.00 ± 1.28 | 23.09 ± 0.32 | 2.91 ± 0.96 | 18.20 ± 0.41 | 1.4521 ± 0.2842 | 13.854 ± 0.426 | 0.1048 |
| *ref: 3D Spatio-Spectral VAE, SAM* | 33.60 ± 0.38 | 27.33 ± 0.93 | 6.27 ± 0.55 | 18.86 ± 0.37 | 0.3728 ± 0.0388 | 15.394 ± 0.024 | 0.0242 |

### 3.6 Probes (P3) for the fusion variants

Each cell shows active-unit fraction / latent-swap Δ / collapsed. P2 `rate_matched=True` holds for every row.

| Dataset | Variant | seed 67 | seed 69 |
|---|---|---|---|
| IIRS | Exp 1 | 0.996 / 1.561 / False | 0.997 / 1.537 / False |
| IIRS | **Exp 2** | 1.000 / **0.010 / True** | 1.000 / **0.009 / True** |
| IIRS | Exp 3 | 1.000 / 1.309 / False | 0.998 / 1.586 / False |
| IIRS | Exp 4 | 1.000 / 0.839 / False | 1.000 / 0.850 / False |
| IIRS | Exp 5 | 1.000 / 0.908 / False | 0.997 / 0.830 / False |
| IIRS | *vae-our-nl* | 0.994 / 1.618 / False | 0.998 / 1.665 / False |
| AVIRIS | Exp 1 | 1.000 / 2.121 / False | 1.000 / 2.235 / False |
| AVIRIS | Exp 2 | 0.890 / 1.326 / False | 0.862 / 1.283 / False |
| AVIRIS | Exp 3 / 4 / 5 | *probes failed* | *probes failed* |
| AVIRIS | *vae-our-nl* | 1.000 / 2.204 / False | 1.000 / 2.196 / False |
| M3 | Exp 1 | 1.000 / 0.214 / False | 0.993 / 0.425 / False |
| M3 | Exp 2 | 1.000 / 0.029 / False | **0.073** / 0.033 / False |
| M3 | Exp 3 | 0.999 / 0.392 / False | 1.000 / 0.419 / False |
| M3 | Exp 4 | 1.000 / 0.167 / False | 1.000 / 0.261 / False |
| M3 | Exp 5 | 1.000 / 0.158 / False | 1.000 / 0.205 / False |
| M3 | *vae-our-nl* | 1.000 / 0.223 / False | 1.000 / 0.367 / False |
| CRIMS | Exp 1 | 0.934 / 0.037 / False | 0.992 / 0.038 / False |
| CRIMS | Exp 2 | 1.000 / 0.044 / False | 0.991 / 0.054 / False |
| CRIMS | Exp 3 | 0.971 / 0.040 / False | 0.999 / 0.037 / False |
| CRIMS | Exp 4 | 1.000 / 0.046 / False | 1.000 / 0.036 / False |
| CRIMS | Exp 5 | 0.927 / 0.036 / False | 0.994 / 0.031 / False |
| CRIMS | *vae-our-nl* | 1.000 / 0.042 / False | 0.999 / 0.045 / False |

The swap Δ is relative to *raw* SAM. On CRIMS that denominator is inflated by the $\pi/2$ pixels, so CRIMS swap values near 0.04 are low for every model and are not evidence of collapse. **P4 SRI is not interpreted here.** Several values are implausible for a relative SAM change: CRIMS Exp 4 6.85 and 1.15, Exp 5 2.23 and −5.98, and *vae-our-nl* on AVIRIS −4.37.

---

## 4. Interpretation and confounds

Confounds come first, because several of them are large enough to change rankings.

### 4.1 `val_loss` is not comparable across variants; the implied multiplier

For Exp 1, 2 and 3 the heartbeat identity holds exactly: $\text{val\_loss}=\text{val\_mse}+\beta\,\text{val\_kld}\,(+\,0.3\,\text{val\_sam})$, with residual below $10^{-4}$ in every heartbeat. For Exp 4 the identity in §1.4 gives an *implied* multiplier $\hat\lambda=(\text{val\_loss}-\text{val\_mse}-\beta\,\text{val\_kld})/\varepsilon$ (*derived*, $\varepsilon=0.08$):

| dataset | seed | epoch | val raw SAM (rad) | val_loss | ELBO = val_mse + β·val_kld | implied λ |
|---|---|---|---|---|---|---|
| IIRS | 67 / 69 | 10 | 0.1876 / 0.1939 | 0.0495 / 0.0492 | 0.0037 / 0.0035 | **0.57 / 0.57** |
| AVIRIS | 67 / 69 | 10 | 0.0331 / 0.0337 | 0.0084 / 0.0078 | 0.0023 / 0.0018 | **0.08 / 0.08** |
| M3 | 67 | 10 | 0.0565 | 0.3711 | 0.3535 | 0.22 |
| M3 | 69 | 10 / 20 / 30 | 0.0729 / 0.0538 / 0.0524 | 0.6037 / 0.8546 / 0.3514 | 0.5871 / 0.8478 / 0.3481 | **0.21 / 0.09 / 0.04** |
| CRIMS | 67 | 10 / 20 / 30 / 40 | 0.4911 / 0.4837 / 0.4817 / 0.4810 | 0.0645 / 0.1683 / 0.2567 / 0.3103 | 0.0028 / 0.0035 / 0.0032 / 0.0032 | **0.77 / 2.06 / 3.17 / 3.84** |
| CRIMS | 69 | 10 / 20 | 0.4861 / 0.4829 | 0.0648 / 0.1676 | 0.0031 / 0.0031 | 0.77 / 2.06 |

Conclusions:

* On CRIMS, val_loss rose **0.0645 → 0.3103** (seed 67) while val SAM was almost flat (0.4911 → 0.4810). The whole rise is $\lambda\varepsilon$. **Do not compare Exp 4 or Exp 5 with anything on `val_loss`.** Compare on SAM-valid, PSNR and MSE.
* Exp 5's loss $\mathcal L_{\text{ELBO}}+\lambda(\overline{v^2}-\bar v)$ goes negative: IIRS final val_loss is −0.2262 (s67) and −0.3058 (s69). It is positive and above the ELBO only on CRIMS, where $\pi/2$ pixels make $v>1$. Neither sign means anything about reconstruction quality.
* The implied $\lambda$ on CRIMS grows by about 1.3 per 10 epochs early on (0.77 → 2.06), then more slowly (+1.11, +0.67) as the cosine LR decays. Meanwhile the violation $\text{SAM}-\varepsilon\approx0.40$ rad stays essentially constant. This fits the dual variable moving at an **LR-bounded rate under AdamW** (§1.4) rather than in proportion to the violation. It is consistent with that reading, but not proof; logging $\lambda$ directly would settle it.
* On M3, the ELBO column is dominated by **KL blow-ups** (val KLD 344–840, §4.6), so the implied $\lambda$ there is a small difference of large numbers. Treat it as indicative only.

### 4.2 The constraint was never a feasible, binding constraint, on any dataset

A constraint only does something different from a penalty when it is **active and satisfiable**. The threshold was one absolute value, $\varepsilon=0.08$ rad $=4.58^\circ$ on *raw* SAM, for all four sensors:

| dataset | val raw SAM reached by Exp 4 (rad) | vs. ε = 0.08 | consequence (implied λ, §4.1) |
|---|---|---|---|
| AVIRIS | ≈ 0.033 | **slack** (≈ 2.4× below) | λ decays 0.30 → 0.08 by epoch 10; the SAM pressure **weakens** below Exp 3's 0.3 |
| M3 | ≈ 0.05–0.07 | **slack** | λ decays to 0.04 by epoch 30 |
| IIRS | ≈ 0.18–0.23 (best of any variant in this grid: 0.120, Exp 3 s69) | **infeasible in practice** | λ grows (0.57 at epoch 10) |
| CRIMS | ≈ 0.48 | **infeasible by construction** | λ grows without bound (3.84 at epoch 40) |

**Why CRIMS is infeasible by construction.** Raw SAM includes every zero-energy pixel at exactly $\pi/2$. On the CRIMS **test** split, `valid_pixel_frac` is 0.498, so 50.2 % of pixels contribute $\pi/2$ and the raw-SAM floor is $0.502\times90^\circ\approx45.2^\circ$. Every model's measured raw SAM is 46.1–47.9°, sitting just above that floor. The **validation** split's raw SAM of ≈0.48 rad implies about 28 % zero-energy pixels (*derived*, assuming valid pixels sit near 2.6° = 0.045 rad: $f\approx(0.48-0.045)/(\pi/2-0.045)$). That gives a training-time floor near 0.44 rad, **more than 5× ε**. No model can satisfy $\text{SAM}\le0.08$ on CRIMS. The dual variable can only keep growing, so Exp 4 on CRIMS is in effect an **increasing-weight SAM penalty**. At the evaluated checkpoints, implied $\lambda$ is ≈3.8 for seed 67 (epoch 39) and between 2.06 and 3.17 for seed 69 (epoch 26, bracketed by its epoch-20 and epoch-30 values). That is **about 7–13× Exp 3's fixed 0.3**.

The same mechanism drives Exp 5 on CRIMS: the $\pi/2$ pixels give $v_i\approx1.5$ rad, which dominates $\overline{v^2}$. The penalty is spent on pixels whose angle no model can change.

Consequence: **no dataset in this grid tests a feasible, binding SAM constraint.** On AVIRIS and M3, Exp 4 is "Exp 3 with a decaying λ". On IIRS and CRIMS it is "Exp 3 with a growing λ". Exp 4's results should be read along that single axis.

### 4.3 Undertraining and the train/eval mismatch must be addressed before any ranking

* **Unequal training length.** With patience 3 and the "neither metric improved" rule, Exp 4 stopped at **12/12 epochs on AVIRIS** and had its best-SAM checkpoint at **epoch 7/8 on IIRS**, barely past the 5-epoch warmup. M3 Exp 1 s67 and M3 Exp 2 s67 stopped at epoch 7 (checkpoint at epoch 4, *during* warmup). Meanwhile IIRS Exp 1 ran the full 60 epochs on both seeds. PSNR on IIRS improves by about 4 dB in epochs 30–40 (§3.1), so the full-test tables partly measure **who was allowed to train longest**.
* **Epoch-10 snapshot.** A matched-length comparison from the heartbeats (val split, raw SAM in rad; M3 Exp 1 and Exp 2 have only seed 69, because seed 67 stopped at epoch 7):

| Dataset | Variant | n seeds | val SAM, raw (rad) | val PSNR (dB) | val SSIM |
|---|---|---|---|---|---|
| IIRS | Exp 1 fixed | 2 | 0.1417 ± 0.0011 | 32.42 ± 0.08 | 0.835 ± 0.000 |
| IIRS | Exp 2 no SAM | 2 | 0.2905 ± 0.0041 | 36.08 ± 0.20 | 0.912 ± 0.003 |
| IIRS | Exp 3 SAM penalty | 2 | 0.1420 ± 0.0017 | 32.73 ± 0.11 | 0.832 ± 0.006 |
| IIRS | Exp 4 PCTF | 2 | 0.1907 ± 0.0031 | 29.52 ± 0.35 | 0.671 ± 0.025 |
| IIRS | Exp 5 U-PCTF | 2 | 0.2443 ± 0.0031 | 30.00 ± 0.46 | 0.699 ± 0.011 |
| AVIRIS | Exp 1 fixed | 2 | 0.0325 ± 0.0002 | 33.61 ± 0.11 | 0.727 ± 0.001 |
| AVIRIS | Exp 2 no SAM | 2 | 0.0492 ± 0.0004 | 33.08 ± 0.03 | 0.722 ± 0.001 |
| AVIRIS | Exp 3 SAM penalty | 2 | 0.0324 ± 0.0008 | 35.38 ± 0.38 | 0.732 ± 0.003 |
| AVIRIS | Exp 4 PCTF | 2 | 0.0334 ± 0.0003 | 34.80 ± 0.15 | 0.736 ± 0.002 |
| AVIRIS | Exp 5 U-PCTF | 2 | 0.0437 ± 0.0007 | 33.27 ± 0.03 | 0.727 ± 0.001 |
| M3 | Exp 1 fixed | 1 (69) | 0.0509 | 31.12 | 0.598 |
| M3 | Exp 2 no SAM | 1 (69) | 0.0910 | 28.17 | 0.651 |
| M3 | Exp 3 SAM penalty | 2 | 0.0475 ± 0.0014 | 30.61 ± 1.75 | 0.625 ± 0.008 |
| M3 | Exp 4 PCTF | 2 | 0.0647 ± 0.0082 | 28.39 ± 0.29 | 0.613 ± 0.021 |
| M3 | Exp 5 U-PCTF | 2 | 0.0678 ± 0.0049 | 29.46 ± 0.21 | 0.642 ± 0.000 |
| CRIMS | Exp 1 fixed | 2 | 0.4891 ± 0.0006 | 31.84 ± 0.20 | 0.634 ± 0.006 |
| CRIMS | Exp 2 no SAM | 2 | 0.5085 ± 0.0002 | 32.89 ± 0.06 | 0.781 ± 0.006 |
| CRIMS | Exp 3 SAM penalty | 2 | 0.4867 ± 0.0004 | 34.93 ± 0.31 | 0.664 ± 0.010 |
| CRIMS | Exp 4 PCTF | 2 | 0.4886 ± 0.0025 | 30.89 ± 0.43 | 0.527 ± 0.002 |
| CRIMS | Exp 5 U-PCTF | 2 | 0.4999 ± 0.0005 | 26.87 ± 1.01 | 0.527 ± 0.007 |

  At a matched epoch, **Exp 4 and Exp 5 are already below Exp 3 on val PSNR on every dataset, and no better on val SAM.** Their poor results are therefore not only an artefact of early stopping. Other orderings are less clean:
  * Exp 3 ≥ Exp 1 on PSNR holds on IIRS, AVIRIS and CRIMS but not on M3, where Exp 1 seed 69 has 31.12 dB against Exp 3's 30.61.
  * On SSIM, Exp 4 slightly exceeds Exp 3 on AVIRIS (0.736 vs 0.732), and Exp 5 exceeds Exp 3 on M3 (0.642 vs 0.625).
  * Absolute values, and the Exp 1 vs Exp 3 gap on IIRS, remain strongly affected by training length.
* **Train/eval mismatch (Exp 4, Exp 5).** Validation, checkpoint selection, test metrics and downstream all see the contracted $\alpha=\tilde\alpha\min(1,\varepsilon/d)$; training never does. One possible cause of Exp 4's large IIRS SSIM and PSNR deficit (0.615 / 27.70 dB vs Exp 3's 0.849 / 34.43 dB) is that **the contraction pushes $\alpha$ toward the spectral stream**, which carries little spatial structure, wherever the two streams disagree by more than 4.58°. If the inter-stream disagreement $d_i$ on IIRS is of the same order as the angle-to-truth (7–11°), most pixels would be contracted. However, $d_i$ was never logged. **This is a hypothesis.** Neither $\alpha$ maps, $d_i$, nor the fraction of contracted pixels were recorded. A related observation: IIRS Exp 4's val SAM got **worse** after the checkpoint epoch (s67: 0.1832 at epoch 7 → 0.2060 at epoch 14; s69: 0.1838 at epoch 8 → 0.2277 at epoch 23) while $\lambda$ was rising. Training continued only because val `mse_final` kept improving.

### 4.4 Exp 2 → Exp 3: the effect of adding SAM (learned gate in both)

Exp 2 never trains on SAM, so its SAM is incidental. The difference Exp 3 − Exp 2 isolates the SAM term:

| dataset | SAM-valid Exp 2 → Exp 3 (°) | Δ | PSNR Exp 2 → Exp 3 (dB) | caveat |
|---|---|---|---|---|
| IIRS | 16.522 → 7.592 | **−8.93** | 36.02 → 34.43 | Exp 2 is **P3-collapsed (INVALID)**, so this delta is against a degenerate model |
| AVIRIS | 2.412 → 1.633 | **−0.78** | 33.62 → 35.81 (+2.19) | |
| M3 | 8.662 → 5.899 | **−2.76** | 29.07 → 30.69 (+1.62) | Exp 2 stopped at 7/10 epochs; s69 has 7.3 % active units |
| CRIMS | 4.491 → 2.760 | **−1.73** | 32.66 → 33.96 (+1.30) | Exp 2 SSIM is higher (0.897 vs 0.872) |

* **The SAM term is essential.** Without it the learned-gate model loses 0.8–2.8° of SAM-valid on the three non-collapsed datasets. On AVIRIS, M3 and CRIMS it does not cost PSNR; it adds 1.3–2.2 dB.
* **IIRS Exp 2 has a collapsed spectral shape.** Swapping latents between patches changes SAM by under 1 % (swap Δ 0.010 / 0.009, below the 0.02 threshold). The raw angle of ~16.5° is the same as the earlier standard-loss baselines (16.5–17.4°). SCC is 0.005 and Q2ⁿ 0.037. Its 36 dB PSNR comes from intensity, not spectral shape. Its **spectral shape is latent-insensitive**, although the output does move with the latent (interpolation path length 0.572). Its tiny downstream PSNR drop (0.24 dB) therefore reflects a decoder whose spectra barely respond to latent changes, not a strength. This fits the history of `standard`-loss cells collapsing spectrally (CLAUDE.md §10, where some collapsed all the way to SAM = π/2). This one is a milder form at about 16.5°.

### 4.5 Exp 1 vs Exp 3: fixed vs learned gate (same loss)

| dataset | SAM-valid Exp 1 / Exp 3 (°) | PSNR Exp 1 / Exp 3 (dB) | SSIM Exp 1 / Exp 3 | note |
|---|---|---|---|---|
| IIRS (means) | 7.224 / 7.592 | 35.91 / 34.43 | 0.891 / 0.849 | Exp 3 s67 undertrained (18 epochs) |
| IIRS (60-epoch runs only) | 7.192, 7.256 / **7.157** | 35.73, 36.08 / **37.70** | 0.889, 0.893 / **0.908** | Exp 3 s69 vs both Exp 1 seeds; stats s69: Exp 3 +1.45 dB (meaningful), SAM-valid Δ 0.0018 rad (negligible) |
| AVIRIS | 1.630 / 1.633 | 34.25 / **35.81** | 0.869 / 0.870 | |
| M3 | 6.306 ± 0.444 / **5.899 ± 0.050** | 30.07 / 30.69 | 0.762 / 0.787 | Exp 1 s67 stopped at epoch 7 |
| CRIMS | 2.850 / **2.760** | 30.38 / **33.96** | 0.741 / **0.872** | |

**When both are trained to comparable length, the learned gate buys fidelity (+1.6 dB on AVIRIS, +3.6 dB on CRIMS, about +1.6–2.0 dB on IIRS at 60 epochs) at equal SAM-valid.** The SAM-valid differences on IIRS and AVIRIS are within ±0.1°, at or below the seed half-ranges and the 0.005 rad (0.29°) effect floor. The fixed blend is not worse at spectral angle; it is worse at spatial detail. This is the expected division of labour: a band-shared $\alpha$ picks *which stream's whole spectrum* to trust per pixel, which affects intensity and texture, while SAM is enforced by the loss.

### 4.6 Seed disagreement and instabilities

| cell | what disagrees |
|---|---|
| IIRS Exp 3 | s67 `[STOP]` at 18 epochs vs s69 `[OK]` at 60: **31.17 vs 37.70 dB**, 8.026° vs 7.157°. This one cell produces the ±3.26 dB half-range. |
| M3 Exp 1 | s67 stopped at 7 epochs (checkpoint at epoch 4) vs s69 at 28: 29.04 vs 31.10 dB, 6.750° vs 5.862° |
| CRIMS Exp 5 | 23.73 vs 28.11 dB (±2.19) |
| M3 Exp 4 | 27.46 vs 29.98 dB; s67 stopped at 16, s69 at 37 |
| IIRS Exp 5 | SAM-valid 11.700° vs 12.857° (±0.58°) |
| M3 Exp 2 s69 | only 7.3 % active latent units |
| CRIMS Exp 3 vs vae-our-nl (stats) | PSNR: s67 Exp 3 +2.92 dB better; s69 vae-our-nl +1.40 dB better. **The sign flips between seeds.** |
| CRIMS Exp 4 vs Exp 3, SAM-valid (stats) | s67 −0.0108 rad, Exp 4 meaningfully better; s69 −0.0028 rad, significant but below the effect floor |

**M3 KL instability.** M3 validation KL reached 117.9 (Exp 3 s69, epoch 20) and 344–840 (Exp 4 heartbeats). On every other dataset's heartbeats it stays at or below 2.2. Final `val_loss` values that are implausible for an MSE of about 0.008 point to the same thing: M3 Exp 1 s67 0.7152, which implies $\text{KL}\approx689$ (*derived*: $(0.7152-0.0088-0.3\cdot0.0571)/10^{-3}$); Exp 3 s69 0.2658. A 4,352-element latent with $\beta=10^{-3}$ is evidently allowed to drift far from the prior on M3. **M3 rankings should not be trusted** until this is understood.

### 4.7 The gate change: Exp 3 vs `vae-our-nl` (the paper's PRISM architecture)

Same backbone, same loss ($\mathcal L_{\text{ELBO}}+0.3\,\text{SAM}$), different gate: 1×1 band-shared here, 3×3 per-band in the paper. The reference rows also come from a different run and code version.

| dataset | SAM-valid Exp 3 / vae-our-nl (°) | PSNR Exp 3 / vae-our-nl (dB) | SSIM Exp 3 / vae-our-nl | stats (same seed, probe sample) |
|---|---|---|---|---|
| IIRS (means) | 7.592 / 7.011 | 34.43 / 39.97 | 0.849 / 0.932 | |
| IIRS (60-epoch Exp 3 s69 vs vae-our-nl s67, s69 at 59 epochs) | 7.157 / 7.074, 6.947 | 37.70 / 39.65, 40.29 | 0.908 / 0.928, 0.936 | s69: vae-our-nl +1.71 dB (meaningful); SAM-valid Δ 0.0032 rad = 0.18° (below the floor) |
| AVIRIS | 1.633 / 1.632 | 35.81 / 35.83 | 0.870 / 0.872 | none (probes failed) |
| M3 | 5.899 / 6.298 | 30.69 / 29.70 | 0.787 / 0.749 | vae-our-nl s67 checkpoint at epoch 5; s67: Exp 3 +2.06 dB and −0.0117 rad (meaningful); s69: both negligible |
| CRIMS | 2.760 / 2.655 | 33.96 / 33.95 | 0.872 / 0.814 | seeds flip on PSNR; SAM-valid: no_difference (s67), negligible (s69) |

**Reading.** On AVIRIS and CRIMS the band-shared 1×1 gate is **indistinguishable** from the paper's gate. PSNR and SAM-valid agree within the seed spread; CRIMS SAM-valid differs by 0.10°, about two half-ranges and below the 0.29° effect floor. On **IIRS the 3×3 per-band gate is better by about 2–2.6 dB PSNR** at matched training length, with SAM-valid equal within 0.2°. That is one seed against two, so it is suggestive rather than settled. On M3 the comparison is confounded by an undertrained reference (checkpoint at epoch 5) and by KL instability. **This grid gives no evidence that the band-shared gate improves spectral fidelity**, which was its motivation (avoiding spectral splicing). SAM-valid is the same as or slightly worse than with the per-band gate on every non-M3 dataset.

### 4.8 Exp 4 and Exp 5 vs Exp 3, and the CRIMS "win"

| dataset | SAM-valid Exp 3 / Exp 4 / Exp 5 (°) | PSNR Exp 3 / Exp 4 / Exp 5 (dB) | effective λ for Exp 4 (§4.1) |
|---|---|---|---|
| IIRS | **7.592** / 11.148 / 12.279 | **34.43** / 27.70 / 29.88 | rising (≥ 0.57), checkpoint at epoch 7/8 |
| AVIRIS | **1.633** / 1.822 / 1.988 | **35.81** / 34.69 / 33.87 | decaying (≈ 0.08 by epoch 10) |
| M3 | **5.899** / 6.830 / 7.384 | **30.69** / 28.72 / 26.79 | decaying (≈ 0.04 by epoch 30) |
| CRIMS | 2.760 / **2.609** / 3.966 | **33.96** / 29.13 / 25.92 | rising (≈ 3.8 at s67's checkpoint, epoch 39; 2.1–3.2 at s69's, epoch 26) |

* **Exp 4 beats Exp 3 on SAM-valid in exactly one place: CRIMS**, by 0.15° (2.609 ± 0.063 vs 2.760 ± 0.062). It is also 0.05° below `vae-our-nl` (2.655 ± 0.047), which is within the half-ranges. It pays **−4.8 dB PSNR and an SSIM of 0.594 vs 0.872**, and the paired test is meaningful for seed 67 only. Given §4.2, this is what a **≈7–13× larger SAM weight** looks like: a fidelity-for-angle trade-off, not a satisfied constraint. Exp 4 also has the highest SCC (0.505) and Q2ⁿ (0.449) on CRIMS. Those small gains are consistent with the same trade.
* Where Exp 4's λ **decayed** (AVIRIS, M3), its SAM-valid is **worse** than Exp 3's, as expected with less SAM pressure.
* **Exp 5 is last or second-to-last on SAM-valid and PSNR on every dataset.** Its CRIMS SID of 0.0826 looks best, but CRIMS SID is strongly affected by the clamp (60.8 % of input pixels have at least one clamped band) and is not ranked.

### 4.9 Downstream (latent robustness and interpolation): do not rank

* The noise is **absolute** ($\sigma=0.5$) and the latents are not normalised to a common scale. With $\beta=10^{-3}$ the scale of each model's posterior means is only loosely tied to $\mathcal N(0,I)$. A model with larger-spread latents is perturbed relatively less by the same $\sigma$. `psnr_drop` therefore mixes robustness with latent scale.
* The models with the best clean reconstructions also show the largest drops: Exp 3 / vae-our-nl lose **14.1–14.4 dB on AVIRIS and 16.9–17.5 dB on CRIMS**, while Exp 1 loses 9.2 / 8.5 dB. Exp 1 is not "more robust" in a sense that matters: its noisy PSNR is higher (AVIRIS 25.89 vs 21.18 dB), but it starts lower and nothing controls for latent scale.
* One batch of 8 patches, one noise draw, and one interpolation path at one pixel. Several half-ranges are as large as the differences. Jag/path differences between fusion variants span up to about 3.5× (IIRS 0.0031–0.0109) but are not interpretable at this power.

### 4.10 Bottom line

**What the grid supports:**

1. **The SAM term is essential.** Adding it to the learned gate (Exp 2 → 3) lowers SAM-valid by 0.8–2.8° on AVIRIS, M3 and CRIMS without costing PSNR. Without it, the IIRS model collapses spectrally (P3 INVALID).
2. **A learned gate beats a fixed 50/50 blend on fidelity, not on angle.** With the same loss and comparable training length, it gains +1.6 dB (AVIRIS), +3.6 dB (CRIMS) and about +1.6–2.0 dB (IIRS, 60-epoch runs), with SAM-valid equal within about 0.1°.
3. **On PSNR and SAM-valid, Exp 3 (the penalty formulation) is the best or tied-best of the four physics-trained variants (Exp 1, 3, 4, 5) on AVIRIS, M3 and CRIMS.** The exception is CRIMS SAM-valid, where Exp 4 is 0.15° lower. On IIRS this holds only for the 60-epoch seed 69. The two-seed means favour Exp 1, because Exp 3 seed 67 stopped at 18 epochs. Exp 2 (no SAM) wins IIRS PSNR and CRIMS SSIM, but with a far worse SAM-valid and, on IIRS, a collapsed latent.

**What the grid does not support:**

1. **"Constraining SAM helps."** The single SAM-valid win (Exp 4 on CRIMS, −0.15° vs Exp 3) costs −4.8 dB PSNR and −0.28 SSIM, is meaningful for one seed only, and arises because the constraint is **infeasible** on CRIMS, which turns it into a growing penalty with an implied λ of ≈2–3.8 at the evaluated checkpoints, 7–13× Exp 3's 0.3. On AVIRIS and M3 the constraint was slack and the result was worse; on IIRS the result was much worse. No dataset tested a feasible, active constraint (§4.2).
2. **"Uncertainty-aware tolerances help."** Exp 5 is worst or second-worst everywhere. Its formulation has a non-detached tolerance, a dual variable that can only grow, a squared-hinge penalty weak below 1 rad, and domination by $\pi/2$ pixels on CRIMS. These are implementation issues as much as a test of the idea.
3. **"The inference-time trust region helps."** It was never trained through, and its effect cannot be separated from that of the constraint without $\alpha$ logging or an ablation that turns it off.
4. **"The band-shared gate is better than the paper's per-band gate."** It is equal on AVIRIS and CRIMS and about 2 dB worse in PSNR on IIRS (one seed), with no SAM-valid advantage anywhere.
5. **Any M3 ranking**, because of KL instability, training runs of 7 epochs, and a +23.5 % rate deviation. **Any AVIRIS significance claim for Exp 3/4/5**, because there are no probes and therefore no stats rows.

For the paper, this grid **neither replaces nor contradicts** the existing PRISM results. It is an ablation that confirms that the SAM penalty and the learned gate are each needed, and finds no benefit from the constrained or uncertainty-aware alternatives *as implemented*.

---

## 5. Recommended next runs

In priority order:

1. **Recover the six AVIRIS probes without retraining.** Read the failure tail in `logs/retrain_pctf_20261118-100225.log` on the lab box. Then run `inference/probes.py --dataset AVIRIS --model <m> --loss physics --seed {67,69} --select sam --ckpt-dir model --out-dir results/probes` for `vae-fusion-{sam-penalty,constrained,uncertainty}`, followed by `verdict.py` and `aggregate.py`. **Do not** re-run `retrain.sh` for these cells: they have no `.done` marker and would be retrained and overwritten.
2. **Remove the training-length confound.** Either use a fixed epoch budget (no early stopping; select the best-SAM checkpoint from all 60 epochs) or set patience ≥ 10 and count it **only after the warmup and the λ transient**. Report epochs-to-best in every table.
3. **Make the constraint meaningful.**
   * Define $\varepsilon$ **per dataset** on **SAM-valid** (mask zero-energy pixels in the training SAM as well). Set it so it is *feasible and active*, for example 0.9× the Exp 3 SAM-valid on that dataset.
   * Give the dual variable its own optimiser (plain SGD dual ascent with a separate step size, $\lambda\leftarrow[\lambda+\eta(\text{SAM}-\varepsilon)]_+$), excluded from weight decay and gradient clipping.
   * Log $\lambda$ every epoch and stop reporting `val_loss` for Exp 4/5 (report $\mathcal L_{\text{ELBO}}$ and SAM separately).
4. **Separate the trust region from the constraint.** Run Exp 4 with the inference contraction **off**. Alternatively, apply the contraction during training too, so it is trained through. Log mean $\alpha$, the fraction of contracted pixels and the mean $d_i$ for every cell.
5. **Fix Exp 5 before re-testing it.** Detach $\varepsilon_i$ (or regularise it). Use a hinge in a well-scaled unit (degrees, or $1-\cos\theta$) so pixels below 1 rad are not under-penalised. Mask zero-energy pixels. Pass the real $\log\sigma^2_s$ in `decode_latents` (or document that downstream uses $\tilde u\equiv1$).
6. **Decouple the gate question from the constraint question.** Add a cell with the **paper's 3×3 per-band gate** under the same retrain (Exp 3 with `AdaptiveGatedFusion`) so that "1×1 band-shared vs 3×3 per-band" is measured within one run and code version. Use ≥ 3 seeds; two seeds and a half-range cannot separate the 0.1–0.2° differences seen here.
7. **M3:** diagnose the KL blow-ups (val KLD 118–840) before using any M3 number, for example with KL warm-up or annealing, or a check of the encoder `logvar` clamp. Also revisit its +23.5 % rate deviation.
8. **Downstream:** scale the noise by each model's latent standard deviation (relative σ). Use ≥ 64 patches and several noise draws and interpolation pairs, so `psnr_drop` and jag/path can be compared across models.

---

## Appendix A: data provenance and corrections

* **Primary data:** `ChatExport_2026-11-20/messages.html` (184 messages, 18.11.2026 10:02 – 19.11.2026 13:14) and the four "(3)" CSVs, the final cumulative exports. Each per-dataset post adds 10 rows (5 variants × 2 seeds), so earlier files are strict subsets. Recon and downstream summaries (seed mean ± half-range) come from the canonical extraction `recon_summary.csv` / `downstream_summary.csv`. Values were spot-checked against the raw CSVs: per-seed recon rows for all fusion cells and `vae-our-nl`, and AVIRIS Exp 1 downstream including the jag/path definition (ratio of seed means).
* **Correction 1: Exp 5 final-epoch training values.** The canonical `pctf_cells_training.csv` mis-parsed **all six** cells whose `[STOP]` tables contain negative losses, likely because of the negative-number parse. Three rows were blank (AVIRIS s67, IIRS s67, IIRS s69). Three silently held the **epoch-10** row instead of the final row (AVIRIS s69, M3 s67, M3 s69). §3.1 uses the final rows read directly from the `[STOP]` messages:

| cell | final epoch | val_loss | val_mse | val SAM (rad) | val PSNR | val SSIM |
|---|---|---|---|---|---|---|
| AVIRIS Exp 5 s67 | 51 | −0.0027 | 0.0005 | 0.0355 | 34.32 | 0.7359 |
| AVIRIS Exp 5 s69 | 35 | −0.0029 | 0.0005 | 0.0363 | 34.11 | 0.7342 |
| IIRS Exp 5 s67 | 18 | −0.2262 | 0.0009 | 0.2060 | 30.88 | 0.7390 |
| IIRS Exp 5 s69 | 19 | −0.3058 | 0.0009 | 0.2413 | 30.95 | 0.7495 |
| M3 Exp 5 s67 | 20 | −0.0086 | 0.0090 | 0.0687 | 28.03 | 0.6445 |
| M3 Exp 5 s69 | 13 | −0.0031 | 0.0087 | 0.0738 | 28.56 | 0.6320 |

  The other 34 cells' final rows match the messages exactly.
* **Correction 2: missing `[START]`.** The export holds 39 `[START]` messages for 40 cells; **IIRS Exp 1 seed 69** has none (its `[HB]` and `[OK]` messages are present). Its batch size is blank in the canonical extraction. It is taken as **32** from `hyperparam-config-IIRS.yaml`, the value every other IIRS `[START]` confirms.
* **Reference rows** (`vae-our-nl`, `vae-our`, the three baselines) are re-exports from the earlier clean-slate grid and were not retrained in this run. The commit and exact protocol of those runs cannot be verified from this export. `ablation_table (3).csv` also holds four `select=mse` rows (M3 `vae-our` / `vae-our-nl`, from the September M3 MSE check). These are excluded from all tables here.
* **Derived quantities:** angle conversions (rad × 180/π); implied λ = (val_loss − val_mse − 10⁻³·val_kld)/0.08; implied M3 KL from val_loss; the CRIMS validation zero-energy fraction estimate (§4.2). All other numbers are copied from the logs or CSVs.

---

## 6. Code and evaluation-setup audit (added 2026-11-20)

Question asked: before Exp 4 and Exp 5 are written off, are the implementation, the maths and the evaluation setup correct?
Method: read `modules/vae_fusion_pctf.py` at 76f17dc, `train/train.py` and `inference/inference.py`; checked each mechanism numerically on CPU with synthetic tensors (no training, no real data); compared validation against training metrics in the Telegram heartbeats.

### 6.1 What is correct

| Item | Check | Result |
|---|---|---|
| Dual-update sign (Exp 4) | $\partial\mathcal{L}/\partial\log\lambda$ for SAM above and below $\epsilon$ | Descent raises $\lambda$ iff SAM $>\epsilon$. Correct. |
| Lagrangian form | $\mathcal{L}=\mathrm{ELBO}+\lambda\,(\mathrm{SAM}-\epsilon)$, with $\lambda$ detached in the primal term | Correct. The logged `val_loss` equals $\mathrm{ELBO}+\lambda\epsilon$ (the SAM terms cancel), which is why it grows with $\lambda$ (section 4.1). |
| Trust-region geometry | $\alpha\cdot\min(1,\epsilon/d)$ keeps the fused spectrum within about $\epsilon$ rad of the spectral stream | Sound as a rule. |
| Checkpoint/metric selection | `train.py` validates under `model.eval()`; early stopping and best-SAM checkpoint use that value | Works as designed, but see F1. |

### 6.2 Findings that make the Exp 4 and Exp 5 comparison unfair

| # | Finding | Evidence | Effect |
|---|---|---|---|
| **F1** | **Train/eval mismatch.** The contraction ran only when `training=False`. The gate was trained without it; early stopping, checkpoint selection and every reported metric used it. | Code: `forward(..., training)` branch. Numerical: with identical weights and input, mean $\alpha$ is 0.574 in train mode and 0.160 in eval mode and MSE rises 0.0072 to 0.0103 (synthetic streams, $d\approx0.29$). Logs, epoch 10, val/train SAM ratio on IIRS: Exp 4 **1.40, 1.43**; Exp 5 **1.75, 1.78**; every other arm **1.03 to 1.07**. val/train MSE ratio: Exp 4 1.75 to 1.92, Exp 5 2.30 to 3.25, others 1.00 to 1.29. On AVIRIS, where stream disagreement is below $\epsilon$ and the contraction is mostly inactive, Exp 4 matches the other arms (SAM ratio 1.04 to 1.06). | The evaluated network is not the trained network. The IIRS and part of the CRIMS deficit of Exp 4/5 is at least partly this. Correlational evidence; section 6.4 gives the test that settles it. |
| **F2** | **One absolute $\epsilon=0.08$ rad for every dataset, on raw SAM.** | Exp 3 reaches SAM-valid 0.029 rad (AVIRIS), 0.048 (CRIMS), 0.125 (IIRS, 60-epoch seed). | AVIRIS: constraint slack, so $\lambda$ falls and SAM pressure vanishes. IIRS: unreachable, so $\lambda$ only grows. CRIMS: zero-energy pixels (about 28% of validation pixels, 50% of test pixels) are exactly $\pi/2$ with zero gradient (checked: grad norm 0), so raw SAM cannot go below about 0.44 rad on validation (the logged val SAM is 0.48) and a 0.08 rad constraint is unsatisfiable regardless of model quality. |
| **F3** | **$\lambda$ is driven by Adam.** | Simulation: with a constant violation of 0.01 or 0.5, $\lambda$ after about 10 epochs is 1.81 in both cases. | $\lambda$ is a fixed-speed ramp that ignores how violated the constraint is; it is not dual ascent. |
| **F4** | **U-PCTF tolerance is not detached.** | `eps_i` has gradient w.r.t. the stream outputs. | The network can lower its own violation by loosening the tolerance. Bounded by `eps_max`, but not intended. |
| **F5** | **U-PCTF penalty is about 40x weaker than Exp 3's.** | Squared violation: $0.3\cdot0.05^2=7.5\times10^{-4}$ against $0.3\cdot0.10=0.03$ for $\lambda\cdot$SAM. | Almost no physics pressure, so Exp 5 behaves like Exp 2. |
| **F6** | **U-PCTF tolerance is centred too low and its multiplier can only rise.** | Typical pixel gives $\tau=\sigma(-1.5)=0.18$, so tolerance is about 0.046 rad, not 0.08. The violation target is exactly 0 and the hinge is never negative. | Heavier eval-time contraction than intended; $\lambda$ cannot fall. |
| F7 | `decode_latents` (used by the latent-noise and interpolation probes) passes `logvar_s=None` for Exp 5. | Code. | Its tolerance differs from the one used in `reconstruct`. Not fixed; the downstream tables for Exp 5 are not comparable to its reconstruction table. |

Not checked or not a defect: the sign of the dual update; the bf16 autocast path (SAM, norm and acos run in float32 under autocast); the early-stopping confound (section 4.3), which affects all arms and is addressed by the rerun using matched patience.

### 6.3 What was changed (hpc branch, uncommitted)

| File | Change |
|---|---|
| `modules/vae_fusion_pctf.py` | F1: contraction in train and eval (`pctf_contract`, default True). F2: constraint on SAM-valid with per-dataset `pctf_epsilon`; reported `sam` stays raw so selection is comparable. F3: projected dual ascent, $\lambda\leftarrow\mathrm{clip}(\lambda+\eta\,(g-\epsilon),0,\lambda_{max})$ with $\eta=0.02$, $\lambda_0=0.3$, $\lambda_{max}=5$, held in a non-persistent buffer. F4: tolerance computed from detached inputs. F5: linear hinge. F6: $\tau$ offset so a typical pixel gets tolerance $\epsilon$; tolerance range $[0.5\epsilon,1.5\epsilon]$; slack $0.1\bar\epsilon$ so $\lambda$ can fall. The old `log_lambda` is kept as a frozen parameter so 2026-11-18 checkpoints still load with `strict=True` (verified). |
| `utils/config.py`, `utils/hyperparams.py`, four `hyperparam-config-*.yaml` | New settings. `pctf_epsilon`: IIRS 0.125, AVIRIS 0.029, CRIMS 0.048, M3 0.103 rad, taken from the SAM-valid the Exp 3 arm reached on each dataset. |
| `train/train.py`, `utils/notify.py` | $\lambda$ is logged each epoch and shown as a `lambda` row in Telegram heartbeats (never logged before). |
| `scripts/retrain.sh` | `--models` filter. |

Checks run on CPU, synthetic data: $\lambda$ rises when the constraint is violated and falls when slack; eval and `no_grad` leave $\lambda$ untouched; train and eval fusion outputs are identical (max difference 0); the U-PCTF tolerance has no gradient and averages 0.100 for a budget of 0.100; losses stay finite over 12 optimiser steps on both models; old-format state dicts load strictly. A first version of the dual step never moved $\lambda$ (a `no_grad` decorator defeated its own grad-mode guard); the checks caught it and it is fixed. **No GPU run and no real data have been used. The fix is unvalidated until the rerun below.**

Choosing $\epsilon$ from Exp 3's result makes the constraint read "be no worse in angle than the penalty arm and spend the rest on fidelity". It is a defensible reading of the original idea, but it is chosen after seeing Exp 3, so say so when reporting.

### 6.4 Commands (lab box)

**(a) Settle F1 without retraining.** Evaluate the existing 2026-11-18 Exp 4/5 checkpoints as they were trained, with the contraction off. If their PSNR/SSIM on IIRS jump towards Exp 3's, F1 explains the deficit.

```bash
git checkout hpc && git pull
mkdir -p results/pctf_nocontract
for DS in IIRS AVIRIS CRIMS; do for M in vae-fusion-constrained vae-fusion-uncertainty; do for S in 67 69; do
  PYTHONPATH=. python inference/inference.py --model $M --dataset $DS --seed $S --ckpt-dir model \
    --set pctf_contract=False --out-json results/pctf_nocontract/${DS}__${M}_s${S}.json
done; done; done
```
Do **not** evaluate the old checkpoints with the new default settings and compare to the logged numbers: the contraction threshold changed (per-dataset $\epsilon$), so those numbers differ from the old ones by construction.

**(b) Rerun with the fixes**, in separate directories so the 2026-11-18 results are untouched. Exp 3 is included so all three arms share the same patience; patience 10 replaces 3 for them.

```bash
bash scripts/retrain.sh --dry-run --models vae-fusion-sam-penalty,vae-fusion-constrained,vae-fusion-uncertainty \
  --datasets IIRS,AVIRIS,CRIMS --patience 10 --ckpt-dir model_pctf_v2 --out-dir results_pctf_v2
# then without --dry-run:
bash scripts/retrain.sh --models vae-fusion-sam-penalty,vae-fusion-constrained,vae-fusion-uncertainty \
  --datasets IIRS,AVIRIS,CRIMS --patience 10 --ckpt-dir model_pctf_v2 --out-dir results_pctf_v2
```
That is 3 models x 3 datasets x 2 seeds = 18 cells (add `M3` to `--datasets` for 24). Longer patience lengthens every cell; the 2026-11-18 cells took 12 min to 3 h each. Watch the `lambda` row in the Telegram heartbeats: it should rise on IIRS until SAM approaches $\epsilon$, hover on AVIRIS, and stay bounded on CRIMS.

**(c) Latent-noise table for the rerun checkpoints** (same flags as the paper models, different checkpoint directory):
```bash
PYTHONPATH=. python inference/inference_final.py --noise-space latent \
  --sigmas 0.01,0.05,0.1,0.3,0.5,1.0 --ckpt-dir model_pctf_v2 \
  --models vae-fusion-sam-penalty,vae-fusion-constrained,vae-fusion-uncertainty \
  --datasets IIRS,AVIRIS,CRIMS --out-dir results/final_latent_noise_pctf_v2
```
(U-PCTF's latent-noise path uses the tolerance without the latent-uncertainty term; see F7.)
