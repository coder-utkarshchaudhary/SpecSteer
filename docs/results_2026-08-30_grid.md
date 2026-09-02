# Ablation grid + inference sweep — results and read

> **Grid:** 2026-08-24 → 2026-08-27 (training), 2026-08-31 (evaluation).
> **Scope:** 4 models × 3 datasets (IIRS, AVIRIS, CRIMS) × {standard, physics} loss × seed axis. M3 dropped.
> **Hardware:** lab box, single NVIDIA RTX PRO 4000 Blackwell (23.4 GB), bf16 autocast, 50 planned epochs, patience 7.
> **Author of this doc:** analysis pass 2026-09-01. Descriptive + hand-adjudicated. The falsification/significance
> layer (`probes.csv`, `stats.csv`, `VERDICT.txt`) is **not yet folded in** — see [§7](#7-what-is-not-in-this-report-yet).

---

## 0. Metric interpretation (for paper writing)

### PSNR is not independent evidence from MSE

`inference/inference.py` computes `psnr = compute_psnr_from_mse(mse)` on the pooled test MSE with `data_range = 1`.
PSNR here is **a deterministic monotonic transform of the reported MSE** — `PSNR = −10·log10(MSE)`. A model cannot be
"close on MSE but beaten on PSNR": the two are the same measurement on a linear vs a log axis.

What looks like "close MSE, lost PSNR" on IIRS is a **scale illusion**. vae-our MSE 0.00043 vs vae-3d 0.00013 reads as
"both ≈ 0" in absolute units, but it is a **3.3× ratio**, which is exactly the 5.3 dB PSNR gap (10·log10 3.3 = 5.2).
The gap is real reconstruction fidelity, not a metric artifact — it is only hidden by reading MSE near zero.

### SAM and (MSE, PSNR, SSIM) measure genuinely different things

| metric | invariances | rewards | penalises |
|---|---|---|---|
| **SAM** | scale-invariant (angle between spectral vectors, per pixel) | correct spectral *shape* / band-to-band ratios | spectral distortion, hallucinated absorption features |
| **MSE / PSNR** | none | correct absolute radiometric level, per band, per pixel | brightness offset, low-frequency spatial error, blur |
| **SSIM** (11×11 windowed, per-band, `data_range=1`) | local luminance offset (partly) | preserved local structure, contrast, texture | spatial smearing, checkerboard/blocky artefacts, contrast loss |

They are not redundant: a model can fix spectral shape (low SAM) while getting overall level or local spatial
structure slightly wrong (higher MSE, lower SSIM), and vice versa. That is the whole reason the ablation reports all
of them.

### Why vae-our wins SAM but loses PSNR/SSIM on IIRS — architectural, not a bug

1. **The spatial stream discards the spatial grid.** `modules/vae_our.py`: `z_spatial` is `(B, latent_dim)` — a
   **single global 256-vector per 64×64 patch**, no H×W. `z_spectral` is `(B, spectral_latent_dim, H, W)` =
   `(B, 4, 64, 64)` — per-pixel but only 4 channels. vae-standard keeps an 8×8×256 spatial *map*; vae-3d keeps an
   8×8×(C/8) *volume*. So vae-standard/vae-3d reconstruct spatial layout from a spatially-localised code; vae-our must
   rebuild all spatial texture from a global vector plus a 4-dim/pixel spectral code. SSIM and PSNR reward exactly the
   spatial structure that vae-our's design throws away in its spatial stream.
2. **The same design is what buys the low SAM.** The 4-channel per-pixel spectral latent + symmetric 1D decoder is a
   clean spectral-shape channel with no 2D/3D convolution mixing neighbouring bands or pixels into it. So spectral
   angle is preserved while spatial fidelity is bottlenecked. This is precisely the ablation's stated hypothesis for
   each model (CLAUDE.md §"Ablation models"): vae-standard "blurs pixel spectra → good PSNR, poor SAM"; vae-1d "great
   SAM, no spatial denoise → poor PSNR/SSIM"; vae-our is supposed to get both.
3. **On IIRS vae-our ≈ vae-1d on every axis** (SAM 0.135 vs 0.133, PSNR 33.3 vs 32.5, SSIM 0.843 vs 0.828). The
   spatial stream is adding almost nothing over the pure pixelwise model there — the "AND" the architecture is for is
   not being achieved. On **AVIRIS it works** (vae-our beats vae-1d and vae-3d on SAM, PSNR *and* SSIM). On CRIMS the
   spatial stream appears to actively hurt (SSIM 0.56).
4. **Seed evidence points to a capacity ceiling, not noise.** All 3 IIRS seeds land at SAM 0.1173 ± 0.0002 (an
   architectural floor) while PSNR varies 32.6–33.9 and val MSE was still dropping at epoch 50. The spectral objective
   is saturated; the spatial reconstruction is capacity- or schedule-limited. A wider spatial decoder, an 8×8 spatial
   latent map instead of a global vector, or more epochs are the levers.

### Is the baselines' "low MSE, high SAM" posterior collapse?

**Partly — and only on the `standard`-loss arm, which is not the main comparison.**

- **`physics`-loss baselines (vae-3d, vae-standard) are NOT collapsed.** vae-3d|IIRS test MSE ≈ 0.00013 is a genuinely
  good reconstruction — a collapsed decoder outputting the mean would give MSE ≈ Var(x) ≈ 3·10⁻³, ~25× higher. Their
  higher SAM is the conv architecture smoothing spectral features while keeping the radiometric level right — the
  designed failure mode, not collapse. Whether their *latent* is informative is a separate question that **P3 (active
  units, latent-swap ΔSAM) and P7 (linear probe R²) answer** — pull those from `probes.csv` before making a collapse
  claim in the paper.
- **`standard`-loss baselines ARE collapsed.** `vae-standard|IIRS|standard` and `vae-3d|IIRS|standard` sit at SAM
  **exactly π/2** with MSE ≈ mean(x²) — textbook decoder collapse (reconstruction orthogonal to input, output near
  constant). But these have no SAM term to prevent it and are not the cells the claim rests on. Do **not** argue "the
  baselines collapse" from these — argue it, if at all, from the physics-loss probe results.

### Recommendation on metrics — keep all four, reframe, add two

- **Do not drop PSNR/SSIM.** The paper's thesis *is* "spectral fidelity without sacrificing spatial fidelity" — you
  need the spatial metrics to state it, and to state honestly where it holds (AVIRIS) and where it does not (IIRS,
  CRIMS-SSIM). Dropping the axes you lose on is the first thing a reviewer checks for.
- **Report as a 2×2**: spectral fidelity {SAM, + a spectral divergence — SID or per-band relative RMSE} vs spatial
  fidelity {PSNR *or* MSE (not both prominently — they are one number), SSIM}. Lead with a Pareto/frontier plot
  (SAM vs PSNR), which is where the dual-stream story actually lives.
- **Replace raw SAM with `sam_valid` + headroom.** Raw SAM has the π/2 contamination (CLAUDE.md §12: CRIMS ~24 %
  sub-ε pixels → hard floor ≈ 0.38 rad unrelated to the model) and a non-zero identity-oracle floor (0.0223 on IIRS).
  Report "% of achievable SAM headroom captured", not the raw angle.
- **Add a downstream/task number.** Reconstruction metrics are proxies for the real goal (LDM-purification readiness).
  Table 4's noise-injection robustness and interpolation smoothness are the genuine differentiators — and there
  vae-3d currently looks *better* on IIRS, so the paper has to engage with that rather than rely on reconstruction
  SAM alone.
- **SSIM is a valid check** (windowed, per-band, unified across scripts per §14.4) — the CRIMS vae-our SSIM 0.56 at
  PSNR 29.7 is a real, informative signal (right pixel values, wrong local structure), not a metric bug. Keep it;
  investigate that cell.

---

## 1. Provenance

| Artifact | Location | Status |
|---|---|---|
| 45 training logs | `logs/train_*_2026082[4-7]*.log` | complete, all 45 cells |
| Training Telegram transcript | `ChatExport_2026-08-30/messages.html` | 45 `[START]`, 34 `[OK]`, 10 `[STOP]` (export cut mid-`vae-3d\|CRIMS\|standard`) |
| 36 reconstruction cells | `logs/inference_*_20260831-16{16..39}*.log` + final aggregate ping in `logs/notify_queue.jsonl` rec 18–19 | complete for the **physics** arm |
| 12 downstream cells | `logs/downstream_*_20260831-1733*.log` | complete, **seed 7** (see bug B1) |
| `probes.csv` / `stats.csv` / `VERDICT.txt` / `results/probes/*.json` | lab box only | **not available here** until ≈2026-09-02 |
| `model/` checkpoints, `data/packed/` | lab box only | not on this machine |

The 16:00–16:03 inference logs are the **synthetic smoke run** (`/tmp/prism_infer_smoke.*` ckpts, `device cpu`) and are
excluded. Only the 16:16–16:39 `cuda` runs against `model/<DS>/` count.

**9 of 45 inference cells are missing**: every standard-loss baseline cell got zero inference because of bug **B1**
([§8](#8-bugs-found)). They are shown as `—` rows below, recovered from **training-log validation metrics** only.

---

## 2. Table 1 — Training (all 45 cells)

`val SAM` / `val MSE` are the best-checkpoint values (`_bestsam` / `_bestmse`); `PSNR@sam` / `SSIM@sam` are validation
metrics **at the best-SAM epoch**. `COLLAPSED` = decoder collapsed to a near-constant output (val SAM ≈ π/2 = 1.5708, or
val MSE ≈ mean(x²)). Wall is total training wall-clock.

### IIRS (C=256)

| model | loss | seed | epochs | status | val SAM @ep | val MSE @ep | PSNR@sam | SSIM@sam | wall | peak GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| vae-our | physics | 7 | 50/50 | OK | 0.11704 @50 | 0.000383 @43 | 34.69 | 0.879 | 1h06m | 14.4 GB |
| vae-our | physics | 42 | 50/50 | OK | 0.11740 @48 | 0.000307 @45 | 35.60 | 0.892 | 1h09m | 14.4 GB |
| vae-our | physics | 1234 | 50/50 | OK | 0.11727 @50 | 0.000289 @50 | 35.88 | 0.898 | 1h06m | 14.4 GB |
| vae-standard | physics | 7 | 50/50 | OK | 0.22864 @49 | 0.000219 @50 | 36.87 | 0.924 | 19m | 1.3 GB |
| vae-standard | physics | 42 | 50/50 | OK | 0.19709 @50 | 0.000145 @50 | 38.77 | 0.940 | 19m | 1.3 GB |
| vae-standard | physics | 1234 | 50/50 | OK | 0.20614 @50 | 0.000165 @50 | 38.20 | 0.933 | 19m | 1.3 GB |
| vae-standard | standard | 42 | 8/50 | **COLLAPSED** | 1.57080 @1 | 0.003193 @1 | 26.03 | 0.167 | 3m | 1.5 GB |
| vae-3d | physics | 7 | 50/50 | OK | 0.20486 @50 | 0.000137 @49 | 38.94 | 0.952 | 4h37m | 14.5 GB |
| vae-3d | physics | 42 | 50/50 | OK | 0.20272 @47 | 0.000134 @50 | 39.06 | 0.953 | 4h37m | 14.5 GB |
| vae-3d | physics | 1234 | 50/50 | OK | 0.20243 @49 | 0.000134 @48 | 39.05 | 0.953 | 4h37m | 14.5 GB |
| vae-3d | standard | 42 | 50/50 | **COLLAPSED** | 1.56934 @50 | 0.003193 @50 | 26.03 | 0.167 | 4h35m | 14.6 GB |
| vae-1d | physics | 7 | 50/50 | OK | 0.11889 @49 | 0.000421 @34 | 34.03 | 0.867 | 1h23m | 14.4 GB |
| vae-1d | physics | 42 | 50/50 | OK | 0.11786 @50 | 0.000429 @29 | 34.02 | 0.869 | 1h23m | 14.4 GB |
| vae-1d | physics | 1234 | 50/50 | OK | 0.11818 @49 | 0.000418 @45 | 34.07 | 0.869 | 1h23m | 14.4 GB |
| vae-1d | standard | 42 | 18/50 | EARLY-STOP | 0.28385 @10 | 0.000357 @11 | 34.01 | 0.817 | 30m | 14.5 GB |

### AVIRIS (C=424)

| model | loss | seed | epochs | status | val SAM @ep | val MSE @ep | PSNR@sam | SSIM@sam | wall | peak GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| vae-our | physics | 7 | 50/50 | OK | 0.02660 @50 | 0.000252 @50 | 36.19 | 0.737 | 1h37m | 12.0 GB |
| vae-our | physics | 42 | 50/50 | OK | 0.02663 @50 | 0.000253 @50 | 36.15 | 0.737 | 1h47m | 12.0 GB |
| vae-our | physics | 1234 | 50/50 | OK | 0.02668 @50 | 0.000252 @50 | 36.19 | 0.737 | 1h37m | 12.0 GB |
| vae-standard | physics | 7 | 42/50 | **FAILED-TRAIN** | 0.09468 @35 | 0.033873 @19 | 15.67 | 0.593 | 17m | 1.1 GB |
| vae-standard | physics | 42 | 15/50 | **FAILED-TRAIN** | 0.09489 @8 | 0.033875 @2 | 15.68 | 0.594 | 6m | 1.1 GB |
| vae-standard | physics | 1234 | 23/50 | **FAILED-TRAIN** | 0.09477 @13 | 0.033855 @16 | 15.56 | 0.591 | 9m | 1.1 GB |
| vae-standard | standard | 42 | 50/50 | OK | 0.04160 @49 | 0.000480 @49 | 33.36 | 0.706 | 21m | 1.2 GB |
| vae-3d | physics | 7 | 50/50 | OK | 0.03265 @49 | 0.000348 @49 | 34.81 | 0.753 | 6h25m | 18.7 GB |
| vae-3d | physics | 42 | 50/50 | OK | 0.03173 @48 | 0.000317 @50 | 35.16 | 0.758 | 6h24m | 18.7 GB |
| vae-3d | physics | 1234 | 50/50 | OK | 0.03267 @47 | 0.000350 @47 | 34.80 | 0.754 | 6h25m | 18.7 GB |
| vae-3d | standard | 42 | 50/50 | **FAILED-TRAIN** | 0.47449 @44 | 0.080922 @19 | 11.00 | 0.513 | 6h19m | 18.8 GB |
| vae-1d | physics | 7 | 50/50 | OK | 0.03097 @50 | 0.000340 @49 | 34.99 | 0.733 | 1h20m | 7.2 GB |
| vae-1d | physics | 42 | 50/50 | OK | 0.02817 @49 | 0.000268 @49 | 35.97 | 0.742 | 1h20m | 7.2 GB |
| vae-1d | physics | 1234 | 50/50 | OK | 0.02827 @50 | 0.000267 @50 | 35.99 | 0.742 | 1h20m | 7.2 GB |
| vae-1d | standard | 42 | 45/50 | EARLY-STOP | 0.04375 @38 | 0.000452 @38 | 33.57 | 0.732 | 1h11m | 7.3 GB |

### CRIMS (C=456)

| model | loss | seed | epochs | status | val SAM @ep | val MSE @ep | PSNR@sam | SSIM@sam | wall | peak GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| vae-our | physics | 7 | 50/50 | OK | 0.48097 @49 | 0.000761 @30 | 32.05 | 0.535 | 35m | 12.9 GB |
| vae-our | physics | 42 | 50/50 | OK | 0.48103 @49 | 0.000757 @45 | 32.12 | 0.531 | 36m | 12.9 GB |
| vae-our | physics | 1234 | 50/50 | OK | 0.48095 @49 | 0.000734 @44 | 32.22 | 0.536 | 35m | 12.9 GB |
| vae-standard | physics | 7 | 50/50 | OK | 0.49750 @48 | 0.001000 @45 | 30.72 | 0.743 | 7m | 1.1 GB |
| vae-standard | physics | 42 | 50/50 | OK | 0.51577 @50 | 0.002510 @29 | 27.08 | 0.739 | 7m | 1.1 GB |
| vae-standard | physics | 1234 | 15/50 | EARLY-STOP | 0.51904 @8 | 0.002874 @5 | 25.63 | 0.723 | 2m | 1.1 GB |
| vae-standard | standard | 42 | 50/50 | OK | 0.51952 @50 | 0.001068 @50 | 30.46 | 0.766 | 7m | 1.3 GB |
| vae-3d | physics | 7 | 50/50 | OK | 0.48200 @40 | 0.000490 @50 | 34.84 | 0.828 | 2h28m | 15.1 GB |
| vae-3d | physics | 42 | 50/50 | OK | 0.48226 @38 | 0.000505 @48 | 34.45 | 0.825 | 2h28m | 15.1 GB |
| vae-3d | physics | 1234 | 50/50 | OK | 0.48208 @36 | 0.000496 @49 | 34.49 | 0.797 | 2h28m | 15.1 GB |
| vae-3d | standard | 42 | 41/50 | EARLY-STOP | 0.50473 @28 | 0.000610 @34 | 33.27 | 0.773 | ~2h | 15.2 GB |
| vae-1d | physics | 7 | 22/50 | EARLY-STOP | 0.55413 @15 | 0.018432 @13 | 17.81 | 0.662 | 11m | 7.2 GB |
| vae-1d | physics | 42 | 21/50 | EARLY-STOP | 0.55378 @14 | 0.017691 @13 | 17.76 | 0.660 | 10m | 7.2 GB |
| vae-1d | physics | 1234 | 27/50 | EARLY-STOP | 0.55347 @20 | 0.017705 @20 | 18.19 | 0.635 | 13m | 7.2 GB |
| vae-1d | standard | 42 | 50/50 | OK | 0.50914 @50 | 0.000746 @50 | 32.46 | 0.769 | 24m | 7.3 GB |

**Training-side observations**

- **vae-3d dominates on validation fidelity** everywhere it trains cleanly (IIRS PSNR@sam 39.0, SSIM 0.95;
  AVIRIS 35.0 / 0.75; CRIMS 34.6 / 0.82) at 25–70× the wall-clock of every other model (IIRS 4h37m vs vae-our 1h07m).
- **vae-our's val SAM is the lowest or tied-lowest** on IIRS (0.117 vs vae-1d 0.118, vae-3d 0.203) and AVIRIS
  (0.0266 vs vae-1d 0.028, vae-3d 0.032). On CRIMS all physics models sit at ≈0.48 val SAM — see the SAM caveat below.
- **Four cells did not train** and must be excluded before any ranking (marked above, and reconfirmed by the P3
  collapse signature applied by hand):
  `vae-standard|IIRS|standard` (π/2 @ep1, stopped 8), `vae-3d|IIRS|standard` (π/2 @ep50),
  `vae-standard|AVIRIS|physics` (val MSE frozen at 0.0339 from ep2, all 3 seeds, PSNR 15.7),
  `vae-3d|AVIRIS|standard` (val MSE 0.081, PSNR 11.0).
- **vae-1d|CRIMS|physics never converged** — early-stops at 21–27 epochs with val MSE ≈0.018, PSNR ≈18. It is a weak
  cell, not a failed one, but its numbers should carry that caveat.
- **vae-1d and vae-standard on CRIMS/standard trained fine** — the standard-loss arm on CRIMS is the healthiest.

---

## 3. Table 2 — Training summary, physics arm (seed mean ± half-range)

Cells that failed to train are omitted. `n=3` seeds unless noted.

| dataset | model | val SAM | val MSE | PSNR@sam | SSIM@sam |
|---|---|---|---|---|---|
| IIRS | vae-our | 0.1172 ± 0.0002 | 0.00033 | 35.4 ± 0.6 | 0.890 ± 0.010 |
| IIRS | vae-1d | 0.1183 ± 0.0005 | 0.00042 | 34.0 ± 0.0 | 0.868 ± 0.001 |
| IIRS | vae-3d | 0.2033 ± 0.0012 | 0.00013 | 39.0 ± 0.1 | 0.953 ± 0.001 |
| IIRS | vae-standard | 0.2106 ± 0.0158 | 0.00018 | 37.9 ± 1.0 | 0.932 ± 0.008 |
| AVIRIS | vae-our | 0.0266 ± 0.0000 | 0.00025 | 36.2 ± 0.0 | 0.737 ± 0.000 |
| AVIRIS | vae-1d | 0.0291 ± 0.0014 | 0.00029 | 35.7 ± 0.5 | 0.739 ± 0.005 |
| AVIRIS | vae-3d | 0.0324 ± 0.0005 | 0.00034 | 34.9 ± 0.2 | 0.755 ± 0.003 |
| CRIMS | vae-our | 0.4810 ± 0.0000 | 0.00075 | 32.1 ± 0.1 | 0.534 ± 0.003 |
| CRIMS | vae-3d | 0.4821 ± 0.0001 | 0.00050 | 34.6 ± 0.2 | 0.817 ± 0.016 |
| CRIMS | vae-1d (not converged) | 0.5538 ± 0.0003 | 0.018 | 17.9 ± 0.2 | 0.652 ± 0.014 |
| CRIMS | vae-standard | 0.5108 ± 0.0110 | 0.0012 | 27.8 ± 2.5 | 0.735 ± 0.010 |

---

## 4. Table 3 — Reconstruction on the held-out test split

**All rows are `select = sam`** (the epoch chosen on best val SAM). The `--select mse` sweep — the fidelity comparison
CLAUDE.md §14.2 preregisters — **was not run**, so the PSNR / SSIM / MSE columns here are *fidelity at the SAM-selected
checkpoint*, not the sanctioned fidelity comparison. `n` is the full test split (IIRS 3084, AVIRIS 1950, CRIMS 369) —
the sweep ran on everything, not the 512-patch default.

### Per-seed (physics arm, from `logs/notify_queue.jsonl` rec 18–19, cross-checked against the individual inference logs)

| dataset | model | seed | test MSE | test SAM (rad) | test PSNR | test SSIM |
|---|---|---|---|---|---|---|
| IIRS | vae-our | 7 / 42 / 1234 | 0.0005 / 0.0004 / 0.0004 | 0.1351 / 0.1347 / 0.1352 | 32.65 / 33.66 / 33.63 | 0.832 / 0.848 / 0.851 |
| IIRS | vae-1d | 7 / 42 / 1234 | 0.0006 / 0.0006 / 0.0005 | 0.1332 / 0.1321 / 0.1325 | 32.53 / 32.49 / 32.61 | 0.827 / 0.828 / 0.829 |
| IIRS | vae-3d | 7 / 42 / 1234 | 0.0001 | 0.2174 / 0.2155 / 0.2151 | 38.88 / 39.00 / 38.99 | 0.954 |
| IIRS | vae-standard (physics) | 7 / 42 / 1234 | 0.0002 / 0.0001 / 0.0002 | 0.2439 / 0.2132 / 0.2214 | 36.67 / 38.45 / 37.88 | 0.918 / 0.938 / 0.929 |
| AVIRIS | vae-our | 7 / 42 / 1234 | 0.0002–0.0003 | 0.0273 | 36.08 / 36.02 / 36.07 | 0.868 |
| AVIRIS | vae-1d | 7 / 42 / 1234 | 0.0004 / 0.0003 / 0.0003 | 0.0315 / 0.0288 / 0.0289 | 34.41 / 35.67 / 35.72 | 0.856 / 0.858 / 0.859 |
| AVIRIS | vae-3d | 7 / 42 / 1234 | 0.0003–0.0004 | 0.0332 / 0.0324 / 0.0333 | 34.14 / 34.59 / 34.15 | 0.848 / 0.854 / 0.848 |
| AVIRIS | vae-standard (physics) | 7 / 42 / 1234 | 0.0337–0.0340 | 0.0995–0.0997 | 14.72 / 14.72 / 14.69 | 0.724 |
| CRIMS | vae-our | 7 / 42 / 1234 | 0.0010–0.0011 | 0.8045 / 0.8045 / 0.8042 | 29.60 / 29.63 / 29.92 | 0.564 / 0.561 / 0.563 |
| CRIMS | vae-1d | 7 / 42 / 1234 | 0.0125–0.0126 | 0.8540–0.8544 | 18.99 / 18.99 / 19.03 | 0.744 / 0.780 / 0.664 |
| CRIMS | vae-3d | 7 / 42 / 1234 | 0.0014–0.0016 | 0.8052 | 28.59 / 27.99 / 28.40 | 0.941 / 0.941 / 0.952 |
| CRIMS | vae-standard (physics) | 7 / 42 / 1234 | 0.0027 / 0.0045 / 0.0048 | 0.8218 / 0.8382 / 0.8331 | 25.69 / 23.50 / 23.17 | 0.888 / 0.880 / 0.854 |

### Seed-aggregated (mean; half-range in parens where > effect floor)

| dataset | model | test SAM | test PSNR | test SSIM | verdict input |
|---|---|---|---|---|---|
| IIRS | **vae-our** | 0.1350 | 33.31 (±0.51) | 0.843 (±0.010) | |
| IIRS | vae-1d | 0.1326 | 32.54 | 0.828 | |
| IIRS | vae-3d | 0.2160 | 38.96 | 0.954 | |
| IIRS | vae-standard (physics) | 0.2262 (±0.015) | 37.67 (±0.89) | 0.928 (±0.010) | seed-unstable |
| AVIRIS | **vae-our** | 0.0273 | 36.06 | 0.868 | |
| AVIRIS | vae-1d | 0.0297 | 35.27 (±0.65) | 0.858 | |
| AVIRIS | vae-3d | 0.0330 | 34.30 (±0.22) | 0.850 | |
| AVIRIS | vae-standard (physics) | 0.0996 | 14.71 | 0.724 | **INVALID (failed train)** |
| CRIMS | **vae-our** | 0.8044 | 29.72 (±0.16) | 0.563 | SSIM anomaly |
| CRIMS | vae-1d | 0.8542 | 19.00 | 0.730 (±0.058) | not converged; SSIM seed-unstable |
| CRIMS | vae-3d | 0.8052 | 28.33 (±0.30) | 0.945 | |
| CRIMS | vae-standard (physics) | 0.8310 (±0.008) | 24.12 (±1.26) | 0.874 (±0.017) | seed-unstable |

### Missing: the standard-loss arm (bug B1)

| dataset | model | loss | status | best proxy (val, from training log) |
|---|---|---|---|---|
| IIRS | vae-standard | standard | no inference | COLLAPSED — val SAM π/2, PSNR@sam 26.0 |
| IIRS | vae-3d | standard | no inference | COLLAPSED — val SAM π/2, PSNR@sam 26.0 |
| IIRS | vae-1d | standard | no inference | val SAM 0.284, val MSE 0.00036, PSNR 34.0, SSIM 0.82 |
| AVIRIS | vae-standard | standard | no inference | val SAM 0.042, val MSE 0.00048, PSNR 33.4, SSIM 0.71 |
| AVIRIS | vae-3d | standard | no inference | FAILED — val MSE 0.081, PSNR 11.0 |
| AVIRIS | vae-1d | standard | no inference | val SAM 0.044, val MSE 0.00045, PSNR 33.6, SSIM 0.73 |
| CRIMS | vae-standard | standard | no inference | val SAM 0.520, val MSE 0.0011, PSNR 30.5, SSIM 0.77 |
| CRIMS | vae-3d | standard | no inference | val SAM 0.505, val MSE 0.00061, PSNR 33.3, SSIM 0.77 |
| CRIMS | vae-1d | standard | no inference | val SAM 0.509, val MSE 0.00075, PSNR 32.5, SSIM 0.77 |

Re-running the fixed script recovers these ([§8](#8-bugs-found)).

---

## 5. Table 4 — Downstream latent probes (seed 7, test split, from `logs/downstream_*_20260831-1733*`)

Experiment 1: encode → add `N(0,σ²)` to the latent → decode, metrics vs the clean reconstruction.
Experiment 2: `z_mix = α·z_A + (1−α)·z_B`, decode, `jaggedness` = mean L2 of the 2nd difference of the decoded spectrum
along α (lower = smoother), `path_length` in spectral space.

### PSNR vs σ

| dataset | model | σ=0 | σ=0.1 | σ=0.5 | σ=1.0 | drop 0→0.5 |
|---|---|---|---|---|---|---|
| IIRS | vae-our | 31.78 | 31.53 | 28.27 | 24.39 | 3.51 |
| IIRS | vae-3d | 37.90 | 37.84 | 36.46 | 33.32 | 1.44 |
| IIRS | vae-1d | 30.74 | 30.65 | 29.16 | 27.07 | 1.58 |
| IIRS | vae-standard | 28.67 | 28.91 | 33.29 | 32.69 | **−4.61** |
| AVIRIS | vae-our | 36.58 | 34.54 | 23.77 | 17.10 | 12.80 |
| AVIRIS | vae-3d | 35.73 | 34.30 | 19.92 | 11.59 | 15.81 |
| AVIRIS | vae-1d | 34.03 | 33.08 | 23.63 | 17.08 | 10.40 |
| AVIRIS | vae-standard | 16.64 | 16.53 | 16.49 | 16.49 | **0.15** |
| CRIMS | vae-our | 31.67 | 29.50 | 18.72 | 13.23 | 12.96 |
| CRIMS | vae-3d | 36.34 | 34.19 | 23.42 | 16.92 | 12.93 |
| CRIMS | vae-1d | 17.94 | 17.94 | 17.63 | 14.54 | 0.31 |
| CRIMS | vae-standard | 20.42 | 20.43 | 22.33 | 19.04 | **−1.91** |

### Interpolation smoothness

| dataset | model | jaggedness ↓ | path_length |
|---|---|---|---|
| IIRS | vae-our | 0.0123 | 0.293 |
| IIRS | vae-3d | **0.0029** | 0.545 |
| IIRS | vae-1d | 0.0210 | 0.313 |
| IIRS | vae-standard | 0.0167 | 0.643 |
| AVIRIS | vae-our | 0.0073 | 1.294 |
| AVIRIS | vae-3d | 0.0064 | 1.496 |
| AVIRIS | vae-1d | 0.0066 | 1.317 |
| AVIRIS | vae-standard | **0.0000** | **0.000** |
| CRIMS | vae-our | 0.364 | 13.90 |
| CRIMS | vae-3d | 0.385 | 16.12 |
| CRIMS | vae-1d | 2.175 | 11.68 |
| CRIMS | vae-standard | 0.719 | 15.75 |

---

## 6. The read

### Adjudication rule (applied to every pair below)

No win or loss is called from a raw delta. Every pairwise gap is judged against:

- **preregistered effect floors** (CLAUDE.md §12): 0.005 rad SAM · 0.5 dB PSNR · 0.01 SSIM;
- **measured nondeterminism floor** (§10.5): 0.0005–0.0036 rad SAM.

A gap inside a floor is a **tie**. Cells that failed to train are **INVALID** and excluded before ranking (the P3
collapse signature is applied by hand here — this is not probe output).

### AVIRIS — the headline claim holds

The claim ("high PSNR *and* low SAM simultaneously") is supported.

| pair | ΔSAM (our better +) | ΔPSNR | ΔSSIM | verdict |
|---|---|---|---|---|
| vae-our vs vae-3d | +0.0057 | +1.76 dB | +0.018 | **vae-our wins all three** (SAM just clears the floor) |
| vae-our vs vae-1d | +0.0024 | +0.79 dB | +0.010 | SAM **tie**; fidelity **win** for vae-our |
| vae-our vs vae-standard | +0.072 | +21.3 dB | +0.14 | vae-standard **INVALID** — do not report as a win |

vae-our has the tightest seeds of any model here (SAM half-range 0.0000, PSNR ±0.03).

### IIRS — the headline claim fails

vae-our buys low SAM at a large fidelity cost.

| pair | ΔSAM (our better +) | ΔPSNR | ΔSSIM | verdict |
|---|---|---|---|---|
| vae-our vs vae-3d | **+0.081** (16× floor) | **−5.65 dB** | **−0.111** | vae-our wins SAM, loses fidelity decisively |
| vae-our vs vae-1d | −0.0024 (vae-1d better, **inside floor**) | +0.77 dB | +0.016 | SAM **tie**; small fidelity win for vae-our |
| vae-our vs vae-standard (physics) | +0.091 | −4.35 dB | −0.085 | same shape as vs vae-3d |

So on IIRS, "PSNR *and* SAM together" is not achieved: vae-3d owns fidelity (PSNR 39.0, SSIM 0.95) and vae-our only
matches the *pixelwise* baseline on SAM. vae-our's own IIRS fidelity is also the least seed-stable of the physics cells
(test PSNR 32.65 at seed 7 vs 33.66 at seed 42).

### CRIMS — report, do not rank on SAM

- **Raw test SAM is 0.80–0.85 rad for every model** against val SAM 0.48–0.55. That split-dependent gap is π/2
  contamination (§12: ~24 % sub-ε pixels on CRIMS give a hard raw-SAM floor near 0.38 that has nothing to do with model
  quality). The ≈0.05 rad spread between models is **not interpretable** without `sam_valid` — deferred to the probe
  output.
- **vae-our takes best test PSNR** (29.7 vs vae-3d 28.3, +1.4 dB — clears the floor) **while collapsing on SSIM**
  (0.563 vs vae-3d 0.945, −0.38). This is not a fidelity trade-off: training val SSIM for vae-our|CRIMS is 0.531–0.536,
  so the model genuinely produces structurally poor reconstructions with low pixel MSE. **Open structural anomaly** —
  flag for investigation, do not average it into a "fidelity" claim.
- n = 369, and vae-1d's test SSIM ranges 0.664–0.780 across seeds. CRIMS carries a small-sample caveat on top of the
  SAM one.

### Cross-cutting

1. **vae-3d is the fidelity ceiling and the cost ceiling.** Where it trains, it wins PSNR/SSIM on every dataset,
   at 4–6 h/run (25–70× vae-our). The ablation's own hypothesis for vae-3d ("param-heavy, collapse-prone") is only
   half borne out: it *is* param-heavy and slow, but it collapses only on the `standard` loss (IIRS), same as
   vae-standard.
2. **The `standard` loss regime is where models collapse.** 3 of the 4 failed-training cells are `standard`-loss
   (`vae-standard|IIRS`, `vae-3d|IIRS`, `vae-3d|AVIRIS`), all with the π/2 or mean(x²) signature. This matches the
   "still open" note in CLAUDE.md §10.5 — it persisted from 5 epochs to 50. With no SAM term, these baselines produce
   nothing usable.
3. **`vae-standard|AVIRIS|physics` is a systematic failure**, all 3 seeds, val MSE frozen from epoch 2. Something about
   AVIRIS (C=424) + the 2D architecture + the physics loss does not optimise. Worth a look before the camera-ready.
4. **Downstream cuts against the Phase-2 thesis.** The stated plan is an LDM operating in this latent space, so latent
   noise-robustness and interpolation smoothness matter. On IIRS, **vae-3d has the most robust and smoothest latent**
   (PSNR drop 1.4 dB vs vae-our 3.5; jaggedness 0.0029 vs 0.0123). vae-our is not the best-behaved manifold on any
   dataset. AVIRIS is a near-tie between vae-our / vae-1d / vae-3d.
5. **`vae-standard`'s latent is dead.** Negative PSNR drop under latent noise on IIRS (−4.6) and CRIMS (−1.9) — noise
   *improves* the output — and exactly `jaggedness = 0.0000, path_length = 0.0000` on AVIRIS. The decoder is ignoring
   the latent. Textbook P3 collapse; confirm against `probes.csv`.
6. **Seed stability is mostly fine.** Physics-arm SAM half-ranges are ≤ 0.0014 for every converged cell — well inside
   the effect floor. The exceptions are all fidelity, all on weak/failed cells: `vae-standard|IIRS|physics`
   (PSNR ±0.9), `vae-standard|CRIMS|physics` (PSNR ±1.3), `vae-1d|CRIMS` SSIM (±0.058).

### One-line summary

**vae-our delivers the "low SAM + high fidelity" claim on AVIRIS, fails it on IIRS (where vae-3d owns fidelity and
vae-our only ties the pixelwise baseline on SAM), and on CRIMS the SAM axis is uninterpretable while vae-our shows a
serious unexplained SSIM collapse.** The verdict is dataset-dependent and currently rests on 1 of 3 datasets.

---

## 7. What is NOT in this report yet

Folded in once the lab-box `results/` tree arrives (≈2026-09-02):

- **P1–P7 PASS/FAIL/INVALID** per cell (`probes.csv`). Whether `probes.py` even ran on 2026-08-31 is unknown by
  construction — it, `verdict.py` and `aggregate.py` write **no log file** (only `inference.py` / `downstream.py` do),
  so the empty `logs/probes_*` glob proves nothing. The 16:39→17:33 gap is consistent with it having run.
- **`sam_valid`** (π/2-excluded SAM) — required before *any* CRIMS SAM ranking, and it will move the IIRS/AVIRIS
  numbers too.
- **Identity-oracle headroom** (a perfect copy scores SAM 0.0223 on IIRS, not 0) — "headroom captured", not raw score.
- **Holm-corrected p-values, paired bootstrap CIs, Cliff's delta** (`stats.csv`).
- **D1–D4 capacity controls** — the "would the baseline win with more params?" question.

When the probe verdicts land: if one contradicts a floor-based call in §6, this report is **amended in place with the
disagreement noted**, not silently overwritten.

---

## 8. Bugs found

### B1 — seed-order bug in `scripts/inference.sh` — **already fixed** (commit `f97c48d`, pulled 2026-09-01)

`SEEDS` is discovered from disk by `find_seeds()` (`modules/registry.py:131`), which returns **ascending** ints →
`[7, 42, 1234]`. The script used `SEEDS[0]` (= 7) as a proxy for the grid manifest's first seed, which is **42**
(`GRID_SEEDS[0]`, `scripts/grid_manifest.sh:34`). Standard-loss baselines are trained at seed 42 only, so all 9
standard-arm inference lookups hit `..._seed7_bestsam.pt`, which does not exist → the 9 missing cells. The probe
`--losses physics` guard and the downstream `--seed` were mis-anchored the same way (downstream ran at seed 7).

**Fix (labmate, `f97c48d`)**: use `GRID_SEEDS[0]` from the sourced manifest instead of `SEEDS[0]` in all three places.
Verified here by `scripts/inference_smoke.sh` (synthetic fixtures, CPU): the standard arm now resolves
`..._seed42_bestsam.pt` and runs.

**To recover the 9 missing cells** (and the standard-arm probes), on the lab box:

```bash
bash scripts/inference.sh                 # full sweep — standard arm now included
# or just the standard arm:
bash scripts/inference.sh --seeds 42
```

### B2 — shell checkpoint guard and Python resolution can disagree — **latent, not yet fixed**

`run_inference` (`scripts/inference.sh:195-217`) tests `-s "${ckpt}"` with a fallback to the pre-seed-axis
`${name}.pt`, but passes only `--seed` to `inference.py`, which **re-resolves independently** via
`resolve_checkpoint()`. If a seeded checkpoint is missing but a stale unseeded one from the 30-epoch grid survives
(`logs/notify_queue.jsonl` rec 6/11 show the old `model/<DS>/vae-our.pt` naming), the shell guard passes and Python
then fails — surfacing as a hard `[ERROR]` / `FAILED` instead of a clean `[skip]`. Worse case: if the fallback *were*
loaded, a 30-epoch v3 checkpoint would be evaluated as a 50-epoch v4 result with nothing in the output saying so.

For the **upcoming full re-run this does not bite** — every one of the 45 seeded checkpoints exists on the lab box
(all 45 training cells saved `_bestsam`/`_bestmse`, including the early-stopped ones). It is worth hardening anyway:
pass the shell-resolved path through as `--ckpt "${ckpt}"` (the flag exists, `inference.py:108`), or drop the
shell-side guessing and let `resolve_checkpoint` be the single source — with the pre-seed-axis fallback disabled
whenever a seed was requested.

### Verification of the upcoming `bash scripts/inference.sh` run

- `bash -n scripts/inference.sh` — clean.
- `scripts/inference_smoke.sh` (real `inference.sh` end-to-end on synthetic shards + checkpoints, CPU) — see
  [smoke result note below]. It exercises recon + probes + downstream + verdict + aggregate and checks every output
  artifact.
- Preconditions on the lab box: `data/packed/{IIRS,AVIRIS,CRIMS}/{train,test}.npy` must exist (hard preflight,
  `inference.sh:117-141`) and `model/<DS>/` must hold the 45 seeded checkpoints. Both were used by the grid, so both
  are present.
