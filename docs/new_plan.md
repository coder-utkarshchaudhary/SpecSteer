# PRISM architecture improvement plan — AAAI 2027, 14-day experiment window

> **Status:** approved 2026-09-01. **Phase 0 (analyze the result files) is a hard gate** — analysis
> comes first, so all iteration *content* below is provisional until Phase 0 runs. Companion doc:
> `docs/results_2026-08-30_grid.md` (the current-grid read this plan responds to).

---

## Context

The 2026-08-30 grid does not give PRISM a clean win. From `docs/results_2026-08-30_grid.md`,
seed-averaged, floor-adjudicated:

- **AVIRIS** — clean simultaneous win (SAM, PSNR, SSIM all clear the floor vs vae-3d; SAM tie / fidelity win vs vae-1d).
- **IIRS** — PRISM SAM 0.135 **ties vae-1d** (0.133, inside the 0.005 floor) and **loses fidelity to vae-3d by 5.6 dB PSNR / 0.11 SSIM**. The paper's "best on all metrics simultaneously" claim fails here.
- **CRIMS** — raw SAM uninterpretable (π/2 contamination); PRISM SSIM **collapses to 0.56** vs vae-3d 0.94 at comparable PSNR — an unexplained structural failure.
- **Downstream** contradicts the thesis: on IIRS vae-3d's latent is *more* noise-robust and smoother than PRISM's.

**Root cause (from the code).** PRISM's spatial stream is a **4096:1 global bottleneck**:
`SpatialBranch.Encoder` does `Flatten(8192) → LazyLinear → z_s ∈ (B, 256)` — a single global vector for
a whole 64×64 patch, no spatial grid. vae-standard keeps an 8×8×256 latent *map*; vae-3d an 8×8×(C/8)
*volume*. PRISM's spectral stream (`SpectralBranch`) is a per-pixel 1D auto-encoder — **operationally
identical to vae-1d** — and the fusion is a fixed global `Linear(2C→C)`. So `recon_final ≈ recon_p ≈ vae-1d`,
with only a weak low-frequency spatial correction from `recon_s`. That is exactly why SAM matches vae-1d
and PSNR/SSIM trail the spatial baselines.

**Goal.** **IIRS is the primary target** — pull PRISM onto the Pareto frontier *on every axis* there
(SAM tied-best, PSNR/SSIM near-best), which is what "spectral **and** spatial fidelity simultaneously"
means concretely. AVIRIS is confirmation (already a clean win). CRIMS is fix-and-win or fix-and-honestly-frame.
Failing the IIRS gate: a defensible fusion / physics-prior contribution. Target venue: **A\* (poster
acceptable); not below A\* main track.**

**Constraints.** Single 24 GB RTX PRO 4000; ~14 experiment days (last 10 of 24 are writing);
**baselines frozen — never re-trained, except** (a) a 1.5× capacity backstop for vae-1d/vae-standard,
(b) a one-time clean CRIMS re-run for all models if CRIMS stays in the paper (CRIMS is the cheap dataset);
**no LDM**.

**User decisions (this session).**
- LDM: none. P6 (noise pass-through / purification) is already the purification evidence.
- Capacity control: keep PRISM param-matched (feasible — see below) + one cheap capacity backstop.
- Fallback: pivot the headline to the **fusion / physics-prior** contribution (primary), Pareto + per-sensor (secondary).
- **IIRS is the dataset the headline claim must win.** CRIMS instability is in scope to fix.

---

## Budget model — why "4 full grids" is the wrong frame

Baselines are done and frozen. Only **PRISM re-trains**: 9 cells (3 datasets × 3 seeds).
Wall from the grid's own logs: IIRS ~1.1 h, AVIRIS ~1.7 h, CRIMS ~0.6 h per seed → **~10.5 h + ~1.5 h eval
≈ 0.75 day per architecture iteration.**

| item | cost |
|---|---|
| Phase 0 — analyze result files | 1 day |
| Iterations 1–3 (PRISM-only, 9 cells each) | ~2.5 days |
| CRIMS clean re-run — all 4 models, fixed training config, once (CRIMS is the cheap dataset: vae-3d 2.5 h, others <0.5 h per seed) | ~0.75 day |
| Iteration 4 — final validation (PRISM + full `--select {sam,mse}` re-inference of frozen baselines) | ~1.5 days |
| Capacity backstop (vae-1d + vae-standard at 1.5× params, 6 cells, seed 42) | ~0.5 day |
| Fold results, redo `docs/results_*`, build paper tables | ~1.5 days |
| **Total** | **~8.25 days**, ~5 days slack |

Slack absorbs failed runs, a 4th iteration, or a CRIMS long run.

---

## Phase 0 — Analyze the result files (BLOCKING, day 1)

Inputs: `results/probes.csv`, `results/stats.csv`, `results/VERDICT.txt`, `results/probes/*.json`,
`results/ablation_table.csv`, `results/downstream_table.csv` (arriving from the lab box).

Answer, per model per dataset:

1. **P3 (collapse), per stream.** Active-unit % and latent-swap ΔSAM for `z_s` vs `z_p` separately. Is the **CRIMS SSIM collapse a dead spatial stream**? Is either stream collapsed on IIRS?
2. **P4 (spatial reliance).** PRISM vs vae-1d SRI. Does PRISM use spatial context *today*, or is it vae-1d in disguise (SRI ≈ vae-1d's ≈ 0)?
3. **P6 (purification NPR).** PRISM vs all, per σ. **This is the paper's core claim.** If PRISM already wins P6 decisively, the whole plan re-weights toward the purification story and away from chasing PSNR.
4. **P5 (spectral inpainting gain).** Does PRISM have a real spectral prior (>10 % over mean-fill)?
5. **P7 (linear probe R²).** Latent = chemistry or nuisance.
6. **P2 (latent budget).** Confirm rate is matched — i.e. the win/loss is architecture, not capacity.
7. **`stats.csv`.** Which pairwise deltas survive Holm + the effect floor. Reconcile against the hand-calls in `docs/results_2026-08-30_grid.md` §6; amend that doc, note any disagreement.

**Output:** rank the failure modes among {spatial latent too weak · spatial stream collapsed · fusion too
crude · spectral latent over-allocated · CRIMS undertrained}. This sets Iteration 1's exact knobs.
If P6 is already a strong PRISM win, escalate the purification framing now.

---

## CRIMS — stabilize, then keep-or-reframe (decided after Phase 0 + Iteration 1)

CRIMS is a mess in the current grid and cannot go into the paper as-is:

| symptom | source |
|---|---|
| vae-1d\|CRIMS\|physics **never converged** — early stop 21–27 ep, test MSE 0.013, PSNR 19 | training logs |
| vae-standard\|CRIMS\|physics seed-unstable — PSNR 25.7/23.5/23.2, seed 1234 stopped at 15 ep | training logs |
| PRISM\|CRIMS **SSIM collapse 0.56** at PSNR 29.7 | inference + train val SSIM 0.53 |
| raw test SAM ≈ 0.80–0.85 for **every** model — π/2 contamination (~24 % sub-ε pixels), split-dependent | CLAUDE.md §12 |
| only **2 561** training patches (uncapped) vs 7 000 elsewhere | pack metadata |

**Two independent problems:**

1. **Eval-side (no retrain):** `probes.py`/`verdict.py` **already compute `sam_valid`** (π/2-excluded) —
   so `probes.csv`/`stats.csv` carry it; Phase 0 just reads it. The gap is `inference/inference.py`
   (the headline reconstruction table) which reports only raw `sam_rad`. Add `sam_valid` there and
   re-run the reconstruction sweep only (~20 min). Raw CRIMS SAM never goes in a ranking.
2. **Train-side (one clean re-run):** the vae-1d/vae-standard non-convergence and PRISM's SSIM collapse
   are undertraining + instability on 2 561 patches. Fix once, for **all 4 models identically** (fairness):
   - `epochs` 50 → 80, `early_stopping_patience` 7 → 12 for CRIMS only;
   - LR warmup (5 ep linear 0 → 1e-3) then the existing cosine;
   - keep batch size, keep the latent/param match.
   Re-run the CRIMS physics arm (4 models × 3 seeds) + CRIMS standard arm (3 models × 1 seed) — ~0.75 day,
   **after PRISM's architecture is frozen** (so PRISM's CRIMS cells use the final architecture).
   Diagnose PRISM's SSIM collapse first from P3: if `z_s` is collapsed on CRIMS, the Iteration-2 free-bits /
   KL-warmup fix applies here; if it is a fusion-conditioning issue at C=456, the Iteration-1 fusion
   redesign should already address it.

**Keep-or-reframe decision (gate, after Iteration 1 + the CRIMS re-run):**

- **Keep** if the CRIMS re-run gives all 4 models a converged, seed-stable result AND PRISM is Pareto-
  competitive (SSIM recovered > 0.85, SAM tied-best on `sam_valid`). → CRIMS is a third supporting dataset.
- **Reframe** if PRISM is fixed but a baseline still wins CRIMS fidelity clearly → CRIMS becomes a
  "harder sensor / where the bottleneck bites" analysis subsection, not a headline row.
- **Drop** only as last resort → paper is IIRS + AVIRIS (2 datasets is thin for an A\* ablation; flag the
  risk to the user before committing).

---

## The "PRISM wins because it is bigger" objection — how it is handled

**Keep PRISM param-matched to ~10.9 M.** This is feasible without limiting the redesign: the current
spatial stream spends **~6 M of PRISM's 10.9 M params on two dense layers** — encoder `Flatten 8192→512`
(~4.2 M) and decoder `256→8192` (~2.1 M). Moving to a conv-projected 8×8 spatial-grid latent **removes
both**, and that headroom pays for the added spatial capacity at equal total params.
`utils/check-model-params.py --solve` (PRISM knobs only) re-audits; baseline widths stay frozen.

- **Backstop (cheap):** train **vae-1d and vae-standard at ~1.5× their params**, physics loss, 3 datasets,
  seed 42 — 6 cells, ~0.5 day. Gives a 2-point capacity slope (D1) for PRISM's *actual* competitors
  (vae-1d on SAM, vae-standard on PSNR). If PRISM's SAM sits off the fitted `metric vs log2(params)` line,
  "bigger" is falsified for the axis PRISM wins on.
- **vae-3d** is defended *by direction*: PRISM is param-matched-or-larger and still **loses** to vae-3d on
  IIRS fidelity, so no size story explains PRISM's SAM advantage over it.
- **D2** (covariate regression on `log2(params)` + `log2(achieved latent)`, all existing cells) and **D3**
  (SAM–PSNR Pareto frontier) remain reported controls — no new runs.

Net: with param-matching held + the D1 backstop for the two light baselines, the objection is closable at
an A\*-review bar.

---

## Architecture changes (staged; final set fixed after Phase 0)

### Iteration 1 — spatial-grid latent + adaptive fusion (the big swing)

1. **Spatial latent: global vector → 8×8 grid.** `modules/SpatialBranch.py`:
   - Encoder: stop the stride-2 Conv2d stack at 8×8 (3 blocks, not 4), then a 1×1 conv to `2·d_s` channels →
     `z_s ∈ (B, 2·d_s, 8, 8)`; drop `Flatten`/`LazyLinear`.
   - Decoder: start from `(d_s, 8, 8)`, `ConvTranspose2d` ×3 → 64×64; drop the `256→8192` `Linear`.
   - `vae_our.py:reparameterize` already chunks on `dim=1` — works unchanged for 4-D.
2. **Rebalance the latent to keep T matched (no baseline re-run).** IIRS T = 16 384:
   `64·d_s + 4096·d_p ≈ 16 384`. Start conservative: `d_p = 3, d_s = 64` (12 288 + 4 096).
   AVIRIS/CRIMS T ≈ 28 672: `d_p` 7→5, `d_s` sized to fill.
   Update `utils/match_latent_rate.py:~90` vae-our formula →
   `s.vae_our_spatial_latent_ch*64 + s.spectral_latent_dim*H*W`; run `--exact --check`.
3. **Fusion: `Linear(2C→C)` → spatially-adaptive gated fusion.** New module in `vae_our.py`:
   2×(3×3 conv, channels-first) over `cat[recon_s, recon_p]` → a per-pixel gate α (and/or a per-pixel
   residual), `recon_final = σ(α·h_s + (1−α)·h_p)`. Small param cost, folded into the match.
   **This is the fallback headline contribution** — keep it clean and ablatable
   (with/without adaptive gate, with/without SAM-on-fusion).
4. **Auxiliary MSE downweight:** `total_mse = 0.5·mse_final + 0.1·mse_spatial + 0.1·mse_spectral`
   (from 0.25/0.25) so each stream can specialize instead of being forced to a standalone full
   reconstruction. Update `paper/draft.md` §3.2.
5. Re-solve PRISM's own width knobs (`reduced_dims`, spatial conv widths) to ~10.9 M via
   `check-model-params.py --solve`; **baselines frozen**.
6. Mirror every config change into the 4 notebooks (CLAUDE.md caveat 6) → `check_notebook_parity.py --execute`.
7. Run: PRISM 9 cells → `inference.sh --select sam` + `--select mse` + probes + downstream (PRISM rows;
   reuse baseline JSON).

**Gate (IIRS-primary — this is the dataset the claim must win):**
IIRS PSNR gap to vae-3d **≤ 2 dB** (from 5.6; target ≤ 1 dB) **and** IIRS SSIM gap to vae-3d **≤ 0.03**
(from 0.11) **and** IIRS SAM **≤ 0.150** (still tied-best with vae-1d) **and** AVIRIS still a clean
3-metric win. CRIMS SSIM ≥ 0.80 is a soft check here — the hard CRIMS fix is the dedicated re-run.
Passing the IIRS gate = PRISM is Pareto-competitive on all three axes on IIRS = the paper's core claim.

### Iteration 2 — conditional refinement (content from It.1 + Phase 0)

- **SAM regressed** → add SAM on `recon_p` + an angle-consistency term
  `|SAM(x, recon_final) − SAM(x, recon_p)|` (spectral stream as the chemistry anchor the fusion cannot
  corrupt); nudge `d_p` up, `d_s` down.
- **PSNR still short** → shift budget (`d_p → 2, d_s → 128`), widen spatial `ConvTranspose` channels, or a
  3×3 spatial-refinement conv on `recon_final`.
- **CRIMS still broken** → free-bits (~0.5 nat/dim) on the spatial-stream KL, or β warmup (0 → 1e-3 over
  10 ep), or +20 epochs for CRIMS only (2 561 patches — undertrained vs 7 000).
- Run PRISM 9 cells + eval.

### Iteration 3 — loss / hyperparameter polish

- Mini-sweep, **IIRS seed 42 only** (~4–6 × 1 h): `lambda_physics ∈ {0.2, 0.3, 0.5}`,
  `beta ∈ {1e-3, 5e-3}`, aux-weight ∈ {0.05, 0.1}. Pick the knee.
- **λ headline stays 0.3** for the cross-model comparison (baselines are at 0.3); PRISM's λ-sensitivity
  becomes its own mini-ablation figure.
- Apply best config → PRISM 9 cells (final architecture) + full eval.

### CRIMS clean re-run (once, after PRISM architecture freezes)

Per the CRIMS section above: all 4 models, CRIMS-only training-config fix (epochs 80, patience 12, LR
warmup), physics arm 4×3 + standard arm 3×1. ~0.75 day. Then apply the keep-or-reframe decision.

### Iteration 4 — final validation (paper numbers)

- PRISM final 9 cells + `inference.sh` full `--select {sam,mse}` across **all** models (baselines reused,
  CRIMS from the clean re-run) + probes + downstream + verdict + aggregate.
- Capacity backstop: vae-1d + vae-standard at ~1.5× params, seed 42 (6 cells).
- These outputs populate every paper table; regenerate `docs/results_2026-08-30_grid.md` → a new
  `docs/results_final.md`.

---

## Files to change (representative)

| file | change |
|---|---|
| `modules/SpatialBranch.py` | Encoder/Decoder: global vector → 8×8 grid latent (**core**) |
| `modules/vae_our.py` | adaptive fusion module; aux-MSE weights; `encode_latents`/`decode_latents` for the new `z_s` shape; optional SAM-on-`recon_p` |
| `utils/config.py` | new `vae_our_spatial_latent_ch`; derived dims in `__post_init__` |
| `utils/hyperparam_configs/hyperparam-config-{IIRS,AVIRIS,CRIMS}.yaml` | `spectral_latent_dim` (d_p), new d_s, re-solved PRISM knobs — **baseline widths untouched** |
| `utils/match_latent_rate.py` (~line 90) | vae-our achieved-latent formula |
| `utils/check-model-params.py` | vae-our param formula / `--solve` restricted to PRISM knobs |
| `notebooks/*.ipynb` ×4 (cell 2) | mirror config → `utils/check_notebook_parity.py --execute` |
| `inference/downstream.py`, `inference/probes.py` | verify noise-inject / interpolate / P3 / P4 handle a 4-D `z_s` |
| `paper/draft.md` §3.1–3.2, §4 | architecture + fusion description; results are stale (says d_p=128, 28 runs, M3) |
| `scripts/grid_manifest.sh` | optional PRISM-only manifest for fast iteration; a CRIMS-only manifest for the clean re-run |
| `train/train.py` (~line 237) | wrap the `CosineAnnealingLR` in a `SequentialLR` with a 5-ep `LinearLR` warmup — no warmup exists today; needed only for the CRIMS re-run |
| `inference/inference.py` | add `sam_valid` (π/2-excluded, `min_energy` from preregistration) to the metrics dict + `ablation_table.csv` — `probes.py`/`verdict.py` already compute it, but the headline reconstruction table does not |

## Verification (per iteration, before the lab run)

- `PYTHONPATH=. python utils/check-model-params.py` → PRISM within ~2 % of 10.9 M; baselines unchanged.
- `python utils/match_latent_rate.py --exact --check` → PRISM within ±10 % of T.
- Shape dry-run (CLAUDE.md §7.1): one batch through `HSI_DualStream_PI_VAE` for C ∈ {256, 424, 456}.
- `utils/check_notebook_parity.py --execute`.
- **Lab only (never this laptop):** `bash scripts/inference_smoke.sh`, then the PRISM run; watch
  `logs/train_vae-our_*` for the π/2 collapse signature by epoch 2.
- Gate metrics vs `docs/results_2026-08-30_grid.md` + the preregistered floors (0.005 rad / 0.5 dB / 0.01 SSIM).

## Fallback ladder (if It.1–2 miss the PSNR gate)

1. **Primary — fusion / physics-prior is the headline.** Contribution = the SAM-bounded spatially-adaptive
   fusion + the §4.2 result that loss-engineering can't repair an entangled encoder. Needs a crisp fusion
   ablation (build it into It.1). Reconstruction parity becomes secondary.
2. **Secondary — Pareto + per-sensor.** PRISM on the SAM–PSNR frontier on every sensor; clean simultaneous
   win on AVIRIS; P6 purification advantage. Drop "best on all metrics."
3. Both target an **A\* poster**. Do not descend to a B-tier main track.

## Out of scope

- No LDM / diffusion training (P6 is the purification evidence).
- No baseline re-training **except** (a) the single 1.5× capacity backstop for vae-1d / vae-standard, and
  (b) the one-time clean CRIMS re-run for all 4 models. IIRS and AVIRIS baseline checkpoints stay frozen.
- No change to preprocessing, packing, splits, or the seed axis (the CRIMS re-run changes only
  epochs / patience / LR-warmup, applied identically to all models).
- No λ change to the cross-model headline comparison.
