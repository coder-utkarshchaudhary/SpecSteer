# PRISM plan — ICLR 2027, 14-day experiment window

> **Status:** revised 2026-09-04 (supersedes the 2026-09-01 AAAI framing). Iteration 1's
> architecture + the new ablation protocol are **implemented and locally verified** — see
> "State of the code" below. Companion doc: `docs/results_2026-08-30_grid.md` (the grid read
> this plan responds to). Everything below that is not marked DONE runs on the lab box.

---

## Venue and clock

- **Target: ICLR 2027, A\* main track** (poster acceptable; not below A\* main track).
- **Abstract deadline: day 14** (≈ 2026-09-18). **Full paper: day 21** (≈ 2026-09-25).
- Days 15–21 are reserved for writing → **14 experiment days**, single 24 GB RTX PRO 4000.
- The headline claim must be locked (final numbers in hand) by **day 10** so the abstract
  states results, not hopes.

## Context — what the 2026-08-30 grid showed

- **AVIRIS** — clean simultaneous win (SAM + PSNR + SSIM).
- **IIRS** — PRISM SAM ties vae-1d and **loses fidelity to vae-3d by 5.6 dB PSNR / 0.11 SSIM**.
- **CRIMS** — PRISM SSIM collapses to 0.56; raw SAM uninterpretable (π/2 contamination);
  vae-1d never converged; vae-standard seed-unstable.
- **Root cause (confirmed in code):** the spatial stream was a **4096:1 global bottleneck** —
  `Flatten(8192) → LazyLinear → z_s ∈ (B, 256)`, one vector per 64×64 patch, no spatial grid —
  plus a fixed global `Linear(2C→C)` fusion. The spectral stream ≈ vae-1d, so
  `recon_final ≈ vae-1d` with a weak low-frequency spatial correction.
- **Phase 0 (result-file analysis) is DONE** (2026-09-04 session): failure ranking =
  spatial latent too weak > fusion too crude > CRIMS undertrained. The falsification suite
  could not adjudicate — P1's trivial floor was miscalibrated (every cell INVALID, including
  vae-3d at 39 dB) and P5/P6/P7 had scaling bugs → suite demoted (see below).

**Goal.** IIRS is the dataset the headline claim must win: SAM tied-best AND PSNR/SSIM at or
near the spatial baselines. AVIRIS is confirmation. CRIMS is fix-and-win or honestly reframe.

---

## State of the code (implemented + verified locally, 2026-09-04)

### Iteration 1 — architecture (DONE, awaiting lab run)

1. **Spatial grid latent** (`modules/SpatialBranch.py`): stride-2 stack stops at 8×8
   (3 blocks), 1×1 conv heads → `z_s ∈ (B, d_s, 8, 8)`. Flatten/LazyLinear and the
   `256→8192` Linear are gone (~6M params of dense compress/expand removed).
2. **Spatially-adaptive gated fusion** (`modules/vae_our.py:AdaptiveGatedFusion`):
   2×(3×3 conv) over `cat[recon_s, recon_p]` → per-pixel, per-band gate α;
   `recon_final = σ(α·h_s + (1−α)·h_p)`. `settings.vae_our_adaptive_fusion=False` restores
   the old Linear fusion — **this is the fusion ablation arm**, and `gate_map()` renders the
   α-map figure. This is also the fallback headline contribution.
3. **Aux-MSE downweight**: reconstruction mix `0.5 : w : w` (w = `vae_our_aux_mse_weight`,
   default 0.1 → 5:1:1), **normalised to sum to 1** so β and λ keep meaning the same thing
   across models. It.3 sweeps w ∈ {0.05, 0.1}.
4. **Latent budget lands EXACTLY on T** now: `64·d_s + 4096·d_p = T`.
   IIRS: d_p=3, d_s=64 → 16,384. AVIRIS/CRIMS: d_p=5, d_s=128 → 28,672. (M3 dropped.)
5. **PRISM at natural width**: `reduced_dims=64` → 4.93M (IIRS) / 5.41M (AVIRIS) /
   5.48M (CRIMS) params — no longer solved to a baseline target.
6. **Verified locally**: forward/backward + encode/decode round-trip for C ∈ {256,424,456,84};
   `match_latent_rate.py --exact --check` = MATCHED; both fusion arms; param audit.

### Capacity protocol (REPLACES param matching — user decision 2026-09-04)

Baselines are retrained at **citable reference widths**, identical across datasets:

| model | reference width | citation |
|---|---|---|
| vae-standard | base_ch 128, n_down 3 | AutoencoderKL, Rombach et al., CVPR 2022 (LDM f8) |
| vae-3d | base_ch 24 (**representative** — swap exact counts if the PDF surfaces) | 3D-CAE, Mei et al., IEEE TGRS 2019 |
| vae-1d | hidden [512, 256, 128] (4 FC layers) | Palsson et al., IEEE Access 2018; Liu et al., IEEE TGRS 2022 |

Resulting params: vae-standard ~23–24M (**larger** than PRISM → defended by direction),
vae-3d ~3.1M and vae-1d ~0.6–0.8M (**smaller** → get capacity points). The size objection is
answered post-hoc: **params column in every table** + **capacity points** — vae-1d and vae-3d
retrained at ≈PRISM's param count, seed 42, IIRS + AVIRIS
(`utils/check-model-params.py --solve-capacity`; IIRS: vae_3d_base_ch 30,
vae_1d_hidden_dims [1788, 894, 447]) + the D2 covariate regression and D3 Pareto frontier.
**Latent rate stays the hard matched control.** Two bonuses expected from reference widths:
the solver-inflated widths are suspected in the vae-standard|AVIRIS|physics failure (frozen
from epoch 2, all seeds) and the vae-1d|CRIMS non-convergence.

### Training-stability fixes (DONE in code)

- 5-epoch linear LR warmup (0.01×→1×) before the cosine, **all cells identically**
  (`train/train.py`).
- CRIMS only: epochs 50→80, patience 7→12 (2,561 training patches vs ~7,000 elsewhere).

### Falsification suite → diagnostics (DONE; user decision 2026-09-04)

- **Removed**: P1 trivial floors (miscalibrated), P5 inpainting, P6 NPR, P7 linear probe
  (scaling bugs), and the suite-level PASS/FAIL/INVALID verdict. Code deleted; git has it.
- **Kept, as diagnostics with no pass/fail semantics**:
  P2 latent-rate audit (fairness certificate), P3 collapse detection (the one exclusion —
  collapsed cells are skipped by the stats), P4 SRI (the before/after architecture figure),
  `sam_valid` (now ALSO in the headline table: `inference/inference.py` + `ablation_table.csv`),
  and the paired statistics (bootstrap CI + permutation + Holm + preregistered effect floors).
- `inference/verdict.py` now writes `DIAGNOSTICS.txt` (no INVALID adjudication).
- Effect floors stay: 0.005 rad SAM · 0.5 dB PSNR · 0.01 SSIM.

---

## Budget model (re-estimated for the new widths)

PRISM ~1h/seed IIRS, ~1.5h AVIRIS, ~1h CRIMS (80 ep) → **9 cells ≈ 0.45 day/iteration**.
vae-3d at base 24 is ~3–4× faster than the old base 45.

| item | cost |
|---|---|
| Iterations 1–3 (PRISM-only, 9 cells each + eval) | ~1.5–2 days |
| Baseline retrain, physics arm (3 models × 3 datasets × 3 seeds, ref widths) | ~1 day |
| Baseline retrain, standard arm (**seed 42 only** — side comparison) | ~0.3 day |
| Capacity points (vae-1d + vae-3d @ ~5M, IIRS + AVIRIS, seed 42) | ~0.4 day |
| Final full eval sweep (`--select sam` + `--select mse`, diagnostics, downstream, aggregate) | ~0.5 day |
| Fold results, `docs/results_final.md`, paper tables | ~1.5 days |
| **Total** | **~5.5–6 days**, ~8 days slack |

## Schedule (day 1 = 2026-09-05)

| days | work | gate |
|---|---|---|
| 1 | Sync repo to the lab box; launch the single chained run (smoke → full grid → capacity points → eval; see "Lab runbook") | smoke clean, grid underway |
| 2 | It.1 eval vs the OLD frozen baselines (still valid for architecture iteration) | **IIRS gate**: PSNR gap to vae-3d ≤ 2 dB (was 5.6), SSIM gap ≤ 0.03 (was 0.11), SAM ≤ 0.150, AVIRIS still a clean win, CRIMS SSIM ≥ 0.80 soft |
| 3–4 | It.2 conditional refinement (SAM-on-recon_p / budget shift / free-bits — see below) | same gate, tightened |
| 5 | It.3 mini-sweep, IIRS seed 42 only: `lambda_physics ∈ {0.2,0.3,0.5}`, β ∈ {1e-3,5e-3}, w_aux ∈ {0.05,0.1}; **λ headline stays 0.3** | knee picked, **PRISM frozen** |
| 6 | Final PRISM 9 cells at the frozen config | |
| 7–8 | Baseline retrain: physics 27 cells + standard arm seed 42 + capacity points | all cells converged; the two old failure cells (vae-std\|AVIRIS, vae-1d\|CRIMS) healthy |
| 9 | Full eval sweep over the NEW grid; diagnostics + downstream + aggregate | |
| 10 | Fold into `docs/results_final.md`; paper tables/figures; **headline locked** | claim survives vs retrained baselines |
| 11–14 | Slack (failed runs, CRIMS retry, 4th iteration); abstract due day 14 | |
| 15–21 | Writing; paper due day 21 | |

**Ordering rule:** freeze PRISM before the baseline retrain so baselines are trained once.
**Risk to watch:** the retrained baselines move the goalposts — vae-3d at 3.1M will likely
lose some fidelity vs its old 10.9M numbers, vae-standard at 24M may gain. The day-10 gate is
judged against the NEW grid; the old grid is only for iteration steering.

## Iteration 2 — conditional content (decided by It.1 results)

- **SAM regressed** → SAM on `recon_p` + angle-consistency term
  `|SAM(x, recon_final) − SAM(x, recon_p)|`; nudge d_p up / d_s down.
- **PSNR still short** → budget shift (IIRS d_p 3→2, d_s 64→96 keeps T exact) or widen
  `reduced_dims`; optional 3×3 refinement conv on `recon_final`.
- **CRIMS SSIM still broken** → free-bits (~0.5 nat/dim) on the spatial-stream KL or β warmup
  (0→1e-3 over 10 ep). The training-config fix (80 ep, patience 12, warmup) is already in.

## Remaining files to change (everything else is DONE — see git log 2026-09-04)

| file | change |
|---|---|
| `paper/draft.md` §3.1–3.2, §4 | grid latent + gated fusion description; capacity-protocol paragraph (reference widths, params column, capacity points; citations in `docs/references.md`); results are stale |

Resolved 2026-09-04 (second pass):
- **Notebooks are retired** (user decision) — they will not be updated or run;
  the pipeline path is `scripts/train.sh` + `scripts/inference.sh` only.
- Grid subsetting: `scripts/train.sh --all --models <csv>` filter (PRISM-only
  iterations / baseline-only retrains) instead of extra manifests.
- Capacity points: `--set KEY=VALUE` overrides in `train/train.py` and
  `inference/inference.py` (train and evaluate with the SAME --set, separate
  `--ckpt-dir`). Solved knobs — IIRS: `vae_3d_base_ch=30`,
  `vae_1d_hidden_dims=[1788,894,447]`; AVIRIS: `vae_3d_base_ch=32`,
  `vae_1d_hidden_dims=[1764,882,441]`.
- Epochs now live in the YAMLs (IIRS/AVIRIS 50, CRIMS 80); `train.sh` no longer
  injects a uniform `--epochs`.
- Baseline citations: `docs/references.md`.
- The whole lab run is one chained command — see "Lab runbook" below.

## Lab runbook — the single command

Resumable by design: `train.sh` skips any slot whose two checkpoints already
exist, so if the chain dies (OOM, power), fix the cause and re-run the SAME
command — completed work is not repeated. Fresh dirs (`model_iclr`,
`results_iclr`) keep the 2026-08 grid untouched for comparison.

```bash
cd <repo> && mkdir -p logs && setsid nohup bash -c '
  export PYTHONPATH=. WANDB_MODE=offline
  python utils/match_latent_rate.py --exact --check && \
  bash scripts/inference_smoke.sh && \
  CKPT_DIR=model_iclr bash scripts/train.sh --all && \
  CKPT_DIR=model_iclr_capacity bash scripts/train.sh --model vae-3d-spatio-spectral --dataset IIRS   --loss physics --seed 42 --set vae_3d_base_ch=30 && \
  CKPT_DIR=model_iclr_capacity bash scripts/train.sh --model vae-1d-pixelwise      --dataset IIRS   --loss physics --seed 42 --set "vae_1d_hidden_dims=[1788,894,447]" && \
  CKPT_DIR=model_iclr_capacity bash scripts/train.sh --model vae-3d-spatio-spectral --dataset AVIRIS --loss physics --seed 42 --set vae_3d_base_ch=32 && \
  CKPT_DIR=model_iclr_capacity bash scripts/train.sh --model vae-1d-pixelwise      --dataset AVIRIS --loss physics --seed 42 --set "vae_1d_hidden_dims=[1764,882,441]" && \
  CKPT_DIR=model_iclr OUT_DIR=results_iclr bash scripts/inference.sh && \
  mkdir -p results_iclr/capacity && \
  python inference/inference.py --model vae-3d-spatio-spectral --dataset IIRS   --loss physics --seed 42 --select sam --ckpt-dir model_iclr_capacity --set vae_3d_base_ch=30                        --out-json results_iclr/capacity/IIRS__vae-3d_capacity.json && \
  python inference/inference.py --model vae-1d-pixelwise      --dataset IIRS   --loss physics --seed 42 --select sam --ckpt-dir model_iclr_capacity --set "vae_1d_hidden_dims=[1788,894,447]" --out-json results_iclr/capacity/IIRS__vae-1d_capacity.json && \
  python inference/inference.py --model vae-3d-spatio-spectral --dataset AVIRIS --loss physics --seed 42 --select sam --ckpt-dir model_iclr_capacity --set vae_3d_base_ch=32                        --out-json results_iclr/capacity/AVIRIS__vae-3d_capacity.json && \
  python inference/inference.py --model vae-1d-pixelwise      --dataset AVIRIS --loss physics --seed 42 --select sam --ckpt-dir model_iclr_capacity --set "vae_1d_hidden_dims=[1764,882,441]" --out-json results_iclr/capacity/AVIRIS__vae-1d_capacity.json
' > logs/iclr_run_$(date +%F_%H%M).log 2>&1 & echo "PID $!  log: logs/iclr_run_*.log"
```

Known risk: PRISM's spatial stream doubled its 64×64 activations (r 32→64); if
`vae-our|IIRS` OOMs at batch 32, set `batch_size: 16` in
`hyperparam-config-IIRS.yaml` (all models — within-dataset fairness holds) and
re-run the same command.

## Verification (per iteration, before the lab run)

- `PYTHONPATH=. python utils/check-model-params.py` → params table (informational).
- `python utils/match_latent_rate.py --exact --check` → PRISM EXACT on T.
- Shape dry-run: one batch through `HSI_DualStream_PI_VAE` for C ∈ {256, 424, 456}.
- **Lab only:** `bash scripts/inference_smoke.sh`, then the run; watch `logs/train_vae-our_*`
  for early-epoch collapse. (Notebooks are retired — no parity step.)
- Gates vs `docs/results_2026-08-30_grid.md` + the effect floors (0.005 rad / 0.5 dB / 0.01 SSIM).

## Fallback ladder (if It.1–2 miss the IIRS gate)

1. **Primary — fusion / physics-prior headline.** The SAM-bounded spatially-adaptive fusion +
   the §4.2 result that loss engineering cannot repair an entangled encoder. The fusion
   ablation arm (`vae_our_adaptive_fusion=False`) and the α-map figure are already built.
2. **Secondary — Pareto + per-sensor.** PRISM on the SAM–PSNR frontier on every sensor;
   clean AVIRIS win. Drop "best on all metrics".
3. Both target an A\* poster. Do not descend to a B-tier main track.

## Out of scope

- No LDM / diffusion training.
- No patch-size change (64 stays): baselines reach the fidelity target at 64, SSIM is 11×11
  windowed, 128px quarters the patch counts (CRIMS → ~600) and ~4×s activation memory.
  Revisit only if post-It.1 SRI saturates while fidelity still trails.
- No LPIPS (user decision 2026-09-04).
- No preprocessing / packing / split / seed-axis changes.
- No λ change to the cross-model headline comparison (PRISM's λ-sensitivity is its own
  mini-ablation).
- M3 stays dropped; its config is kept coherent for the audit scripts only.
