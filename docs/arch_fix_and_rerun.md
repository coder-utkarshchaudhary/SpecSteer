# Architecture fix + re-run spec (2026-09-06)

Companion to `~/.claude/plans/hey-so-i-used-witty-stroustrup.md`. This is the
in-repo, executable version: what the bug is, what changes, how you check the
fix, and how you launch the grid.

---

## 1. The bug — `vae-our`'s fused output cannot leave [0.5, 0.73]

Introduced by commit `931c0fd` ("major fix to architecture"). `modules/vae_our.py`
`AdaptiveGatedFusion.forward`:

```python
fused = alpha * h_s + (1 - alpha) * h_p   # convex combo of the two branch recons
return torch.sigmoid(fused)               # <-- sigmoid applied to an already-[0,1] value
```

`fused` is a convex combination of `recon_s` and `recon_p`, so it is bounded by
their range. The aux-MSE terms (`vae_our.py` `loss_terms`) train both branch
reconstructions toward the data `x ∈ [0, 1]`:

```python
mse_spatial  = self.mse_loss_fn(recon_s, x)
mse_spectral = self.mse_loss_fn(recon_p, x)
```

So the fusion receives values in `[0, 1]`, treats them as **logits**, and
`sigmoid([0,1]) = [0.5, 0.731]`. A dark pixel (`x → 0`) needs the fusion input at
`−∞`, while the aux MSE simultaneously pins that same value near 0. Deadlock —
the loss is flat from epoch 10 in `arch-change-run-logs.txt`, PSNR 9–16 dB on
every dataset against 31–38 dB for every baseline on the same data.

Every baseline does the correct thing — `vae_standard.py:102`, `vae_3d.py:158`,
`vae_1d.py:105` all `sigmoid` a **raw** decoder output. PRISM's two branch
decoders (`SpatialBranch.py`, `SpectralBranch.py`) are the only decoders in the
repo ending on a bare conv, and that bare output is then double-squashed by the
fusion.

The old `LinearFusion` hid this because `Linear(2C→C)` can rescale `[0,1]` inputs
into wide logits. Removing that projection exposed it.

Verified numerically (fusion optimised with `recon_s`/`recon_p` as free
parameters — an upper bound on any encoder): shipped head tops out at **11.5 dB**;
sigmoid removed → **66 dB**.

---

## 2. The fix (adds zero parameters — existing checkpoints stay loadable)

Match every baseline: sigmoid on each branch decoder, plain convex blend in the
fusion.

| file | change |
|---|---|
| `modules/SpatialBranch.py` | `Decoder.forward`: `return torch.sigmoid(x)` (was `return x`) |
| `modules/SpectralBranch.py` | `Decoder.forward`: `return torch.sigmoid(x)` (was `return x`) |
| `modules/vae_our.py` | `AdaptiveGatedFusion.forward`: drop the outer `torch.sigmoid` — return the convex blend directly |
| `modules/vae_our.py` | docstrings (lines ~13, 22, 37, 59): the fused output is now `alpha*h_s + (1-alpha)*h_p`, no outer sigmoid |

`LinearFusion` is **not** changed: `Linear` emits logits, so its own sigmoid is
still correct. Both fusion arms now consume identical `[0,1]` branch
reconstructions, which is what makes the fusion ablation clean.

`vae-3d` fix (grad-accumulation vs. batch drop) is decided from the lab logs —
see the plan file §"Finding 2". Not in this doc yet.

---

## 3. Post-fix smoke — `bash scripts/run.sh`

`run.sh` runs three independent checks (~30 min, mostly CPU + one short GPU
train). Read the tail of each section it prints.

| check | what it proves |
|---|---|
| `match_latent_rate.py --exact --check` | latent-rate fairness control still MATCHED |
| `check-model-params.py` | param count **unchanged** by the fix (no architecture drift) |
| `inference_smoke.sh` | `inference.py` / `probes.py` / `downstream.py` still run through the changed model (decode path, gate map) |
| 5-epoch `vae-our\|IIRS\|physics` train on real data | the fix actually restores reconstruction |

### PASS / FAIL — check these by hand against `logs/fixcheck_vae-our_IIRS.log`

**PASS (launch the grid):**
- `match_latent_rate` prints `MATCHED` / exit 0
- `check-model-params` PRISM param count within ±0 of the pre-fix value
  (fix adds no params; a moved count means something else changed)
- `inference_smoke.sh` prints `SMOKE PASSED`
- 5-epoch probe: **val PSNR ≥ 22 dB at epoch 5 and rising across epochs 3→5**,
  **val SAM ≤ 0.22 at epoch 5 and falling**
  (reference: the broken run was 9.03 dB / 0.31 rad and *flat*; `vae-standard`
  reaches 35.9 dB by epoch 10; the pre-It.1 architecture reached 33.3 dB at 50)

**FAIL (stop, send the log, do not launch):**
- 5-epoch probe val PSNR still < 20 dB, or PSNR not increasing → a second bug in
  the grid-latent spatial branch; cheaper to find now than after a 13 h grid
- `inference_smoke.sh` fails → the decode / gate-map path broke
- param count moved → unintended architecture change

---

## 4. Full grid — `bash scripts/run_entire_grid.sh`

Launch only after the smoke PASSES. Writes to **fresh** dirs so the broken
`model_iclr/` run from `scripts/run.sh` is neither reused nor lost:

- checkpoints → `model_fix/<DS>/`
- results → `results_fix/`

**Grid: 2 seeds {69, 67}, both `physics` and `standard` arms, all 3 datasets.**
`run_entire_grid.sh` calls `scripts/train.sh --all` once per seed with
`GRID_SEEDS=<seed>` — because `train.sh` gives *every* config the seeds in
`GRID_SEEDS`, a single-seed pass runs the physics AND standard arms at that seed.
Two passes → both arms at both seeds. 3 datasets × 7 configs × 2 seeds = **42
cells**. Resumable: a re-run skips any cell whose two checkpoints already exist.

Then the eval sweep, once per seed (`GRID_SEEDS=<seed>` makes the standard-arm
gate in `inference.sh` fire for that seed), for both checkpoint selections
(`--select sam` and `--select mse`).

Est. wall on the TITAN RTX: ~13 h/seed grid + ~2 h/seed eval ≈ **30 h total**.
Override dirs with `CKPT_DIR=... OUT_DIR=... SEEDS="69 67" bash scripts/run_entire_grid.sh`.

---

## 5. Reading the results

- **PSNR is `−10·log10(MSE)`** — one number, not two. "close MSE, lost PSNR" is
  impossible (`docs/results_2026-08-30_grid.md` §0).
- **CRIMS: read `sam_valid`, not raw SAM.** ~24 % of CRIMS pixels sit below the
  SAM epsilon and each contributes exactly π/2, a ~0.377 rad raw-SAM floor
  unrelated to model quality (CLAUDE.md §12). Every CRIMS model landing at
  0.48–0.51 raw SAM is that floor, not collapse.
- **Day-2 gate (vs. the OLD grid, for iteration steering):** IIRS PSNR gap to
  `vae-3d` ≤ 2 dB (was 5.6), SSIM gap ≤ 0.03 (was 0.11), SAM ≤ 0.15; AVIRIS still
  a clean simultaneous win; CRIMS `sam_valid` interpretable and SSIM ≥ 0.80 soft.
- Effect floors (a smaller gap is not a result): **0.005 rad SAM · 0.5 dB PSNR ·
  0.01 SSIM** (CLAUDE.md §12); same-seed nondeterminism floor 0.0005–0.0036 rad
  (§10.5) — which is why the grid runs ≥2 seeds.
