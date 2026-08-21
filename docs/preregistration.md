# Preregistered analysis plan — HSI VAE ablation

**Registered 2026-08-21**, before any probe was run, against checkpoints trained
at a matched 64:1 latent rate. Machine-readable form:
`inference/preregistration.yaml`, which `inference/probes.py` reads at startup
and refuses to run without.

Changing a threshold after seeing results is legitimate only as an explicit,
dated, reasoned edit to that file — which git records. This document explains
*why* each number is what it is.

---

## Why preregister at all

The suite exists to try to falsify the paper's claim, not to confirm it. That
only works if the bar is set before the results are visible; otherwise a
threshold silently becomes "whatever the proposed model happened to clear". Two
of the rules below (P1's floors, P6's claim) would be trivially satisfiable if
written afterwards.

---

## Sampling

512 patches per cell, seeded, stratified across scenes.

A full sweep is 28 cells × 7 probes and several probes are per-patch, so an
uncapped run is unbounded. 512 paired samples is ample for the effect sizes at
stake, and **the confidence intervals are computed on exactly this subset and
report the real n** rather than implying the full split. Stratification matters
because scenes differ several-fold in patch count; a flat draw would
over-represent the large ones and P7's scene probe would then be measuring that
imbalance rather than the latent.

---

## P1 — Trivial-predictor floors

**Question.** Is the model better than predicting the mean?

**Baselines.** Global train mean, per-scene mean, train-fold mean (leakage-free
by construction), the patch's *own* spatial-mean spectrum, and 1000 random
draws (uniform and train-matched normal).

`mean_patch` is the one that bites: it needs no training set at all. A model
that cannot beat it has learned nothing about this data, however good its
absolute PSNR looks.

**Threshold.** Beat the best floor by **≥1.0 dB PSNR, ≥10 % relative SAM,
≥0.02 SSIM**, and sit above all 1000 random draws (p < 0.001).

1.0 dB is roughly the smallest PSNR gap that survives visual inspection on
hyperspectral reconstructions; 10 % relative on SAM rather than absolute because
SAM's scale differs by an order of magnitude between sensors.

**Two corrections without which the numbers mislead.**

- **SAM has a non-zero floor.** A *perfect copy* scores 0.0223 on IIRS, not 0,
  because of the epsilon in its norm. The identity oracle is therefore the real
  ceiling, and the suite reports *headroom captured* =
  `(model − floor) / (oracle − floor)`.
- **SAM has a π/2 contamination.** A pixel with spectral energy below the
  epsilon gets a norm dominated by it, `cos_sim → 0`, and contributes **exactly
  π/2 whatever the model predicted**. CRIMS has ~24 % such pixels, giving its
  raw SAM a hard floor near 0.377 rad unrelated to model quality. `sam_valid`
  excludes them and is the cross-dataset comparable number.

---

## P2 — Latent budget / rate control

**Question.** Is the win capacity rather than architecture?

Reconstruction quality is bounded by rate almost by definition. Before the
64:1 matching, latent budgets varied **512×** within a dataset — vae-standard at
1024:1 against vae-our at 2:1 (and on M3, vae-our's latent was 1.52× *larger*
than its input). Any reconstruction comparison across that spread is
uninterpretable.

**Threshold.** Every model within **±25 %** of the 64:1 target. Cells outside
are reported as rate-unmatched and excluded from the headline comparison.

±25 % is set by what is achievable, not by preference: the per-pixel models can
only hit multiples of 4,096 elements, and 4,096 does not divide M3's 5,376
target, so M3 lands at −19 %/−24 %. Anything tighter would be unsatisfiable.

**Secondary control.** Regress each metric on `log2(achieved latent)` and
require vae-our's residual to exceed **one pooled SD** — a win must beat what
its rate alone predicts.

**Per-branch MSE.** `vae-our`'s `total_mse = mse_final + 0.5·mse_spatial +
0.5·mse_spectral`, and its 256-dim whole-patch spatial bottleneck cannot
reconstruct a full cube. A large `mse_spatial` is an artifact of the auxiliary
weighting, not evidence about reconstruction quality, so the three terms are
reported separately.

---

## P3 — Posterior collapse

**Question.** Is the latent used at all?

The v2 logs show `vae-standard|IIRS|standard` and `vae-3d|IIRS|standard` with
SAM = 1.5708, KLD ≈ 0.0003 and MSE pinned at mean(x²) — decoder output ≈ 0,
latent ignored. Scoring those as if they were models would be wrong.

**Threshold.** Active-unit fraction < **1 %** (per-dim KL > 0.01 nats), **or**
latent-swap ΔSAM < **2 %** ⇒ `INVALID`.

The latent swap is the decisive test: decode patch *i*'s code in patch *j*'s
slot. If the output barely moves, the decoder is producing a constant. KL alone
can mislead, hence two independent signals.

Only the `standard` variants die, which is itself a result: SAM's
scale-invariant gradient is what rules out the all-zero solution. **"The
baseline collapses without a physics term" supports the thesis** and is reported
as a finding, not hidden.

---

## P4 — Spatial-reliance shuffle

**Question.** Does the model use spatial context, or is it pixelwise in disguise?

Permute the H×W grid, keeping each pixel's spectrum intact. A model that only
sees one pixel at a time cannot notice.

**Threshold.** `SRI = (SAM_shuffled − SAM_intact) / SAM_intact < 0.02` ⇒ no
spatial information used.

**Positive control.** `vae-1d-pixelwise` is *exactly* permutation-equivariant,
so its two scores must match to **1e-6**. Any larger deviation is a **bug in the
probe** — almost certainly the permutation applied to the input but not the
target — and is reported as `PROBE_BUG`, never as a finding about the model.
This is the probe's own self-test.

**This is the probe most able to falsify the paper.** If vae-our's spatial
stream scores SRI ≈ 0, the dual-stream claim is empirically empty.

---

## P5 — Spectral band-masking inpainting

**Question.** Is there a spectral prior, or does the model copy the input?

Zero a contiguous 10 % band block, score only those bands, averaged over 5
positions. A pass-through autoencoder reproduces the zeros.

**Threshold.** ≥**10 %** relative improvement over filling with the band-wise
train mean.

Below that, "hallucination-free purification" has no mechanism behind it: a
model with no spectral prior cannot purify anything.

---

## P6 — Noise pass-through / purification

**Question.** Does it actually purify? This is the title's claim.

Noise on the **input** (`downstream.py` perturbs the *latent* — a different
question), at σ ∈ {0.01, 0.05, 0.10} × data SD.

`NPR = ‖D(E(x_noisy)) − x_clean‖ / ‖x_noisy − x_clean‖`

**Thresholds.** NPR < **0.90** denoises · 0.90–1.10 pass-through · > 1.10
actively harmful.

**Preregistered claim, recorded so it can be falsified:** *vae-our attains the
lowest NPR at every σ on every dataset.* If it does not, that is the finding.

---

## P7 — Linear probe on frozen latents

**Question.** Does the latent encode chemistry, or just which scene it came from?

Two linear probes on the same features: continuum-removed absorption depth
(physics) and scene identity (nuisance).

**Thresholds.** Physics R² ≥ **0.5** ⇒ the latent represents something
meaningful. Scene accuracy > **0.8** with physics R² < 0.5 ⇒ the latent is
mostly acquisition nuisance — "nonsensical" in the sense the suite is asking
about.

**Dimensionality is equalised first.** Every latent is PCA-projected to 256
dims before fitting, because a wider latent hands a linear model more columns
and would otherwise look like richer content — the same confound P2 controls.

**Band indices, not nanometres.** The packed shards carry no wavelength vector,
and the probes read only frozen artifacts, so features are located at the
deepest minima of the train-mean continuum-removed spectrum by index. Adding
wavelengths to `pack.py`'s sidecar later would let these be *named* (1900 nm
hydration, and so on) without changing the maths.

Controls: raw pixels (upper bound) and shuffled latents (chance floor).

---

## P8 — Statistics

Paired throughout: every model sees the same patches, so the quantity is the
per-patch difference. Pairing removes patch-to-patch variance, which on HSI is
enormous — a shadowed patch is hard for everyone.

- Paired bootstrap, 10,000 resamples → 95 % CI on Δmetric
- Paired permutation test (sign-flip, the exchangeable unit under the null)
- Holm–Bonferroni across the 6-member model-pair family within each dataset
- Effect sizes: paired mean difference and Cliff's delta

**Minimum meaningful effect: ΔPSNR ≥ 0.5 dB, ΔSAM ≥ 0.005 rad, ΔSSIM ≥ 0.01.**

This is doing the real work. With hundreds of paired patches essentially any
non-zero difference reaches p < 0.05 — a 0.05 dB gap at n = 3,084 gives
p = 0.0005. Results that clear significance but not the effect floor are
labelled `significant_but_negligible` and are **not** claimed as wins.

---

## Decision summary

| Verdict | Meaning |
|---|---|
| `PASS` | valid cell, all probes cleared |
| `FAIL` | valid cell, one or more probes failed (listed) |
| `INVALID` | not a usable result — collapsed, or below a trivial predictor |
| `PROBE_BUG` | the probe's own control failed; fix the probe, ignore the number |

A claimed win requires: cell `VALID` **and** rate matched **and** the pairwise
difference significant after Holm **and** above the effect floor.
