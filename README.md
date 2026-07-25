# specsteer — Dual-Stream Physics-Informed VAE for HSI

Full pipeline description lives in [CLAUDE.md](CLAUDE.md). This README covers
the day-to-day launch commands only.

## Overnight training on two 24GB workstations

The 28-run ablation grid is split across two lab boxes that share the same
external drive.

- **Box A (Utkarsh) — IIRS + M3, 14 runs**
- **Box B (teammate) — AVIRIS + CRIMS, 14 runs**

### One-time setup on each box

```bash
cd ~/prism                           # or wherever the repo lives
git pull
ls "/media/yashdeep/New Volume 21/UTKARSH_CHAUDHARY_prism/data/processed"
# expect: IIRS  AVIRIS  m3  crims
[ -d .venv ] && source .venv/bin/activate
python utils/check-model-params.py   # sanity: prints param counts
```

If the drive is mounted at a different path, export `PRISM_DATA_ROOT` before
launching:
```bash
export PRISM_DATA_ROOT=/mnt/prism/data/processed
```

### Launch

Box A:
```bash
bash scripts/run_overnight.sh --datasets IIRS,M3 --epochs 100
tail -f logs/overnight_$(hostname)_*.log
# safe to close the ssh window after the "PID:" line prints
```

Box B:
```bash
bash scripts/run_overnight.sh --datasets AVIRIS,CRIMS --epochs 100
tail -f logs/overnight_$(hostname)_*.log
```

### In the morning

```bash
# Check that the run finished; grep for the summary banner.
tail -50 logs/overnight_$(hostname)_*.log

# Count checkpoints — 14 per box, 28 across both.
find model -name '*.pt' | wc -l

# Inspect the split:
find model -name '*.pt' | sort
```

### Managing an in-flight run

```bash
bash scripts/run_overnight.sh --status   # is it alive?
bash scripts/run_overnight.sh --stop     # SIGTERM + SIGKILL fallback
```

### What "robust" means here

- Each run is wrapped so **one crashed run doesn't kill the other 13**.
- A failed run is **auto-retried once** (5 s cooldown) before being marked
  `FAILED`. Transient CUDA blips typically clear on the retry.
- On re-launch, any run whose checkpoint file already exists is **skipped**
  — the grid is idempotent, so `git pull && bash scripts/run_overnight.sh
  --datasets IIRS,M3 --epochs 100` picks up where a crashed session left off.
- No epoch-level resume — a run killed mid-training restarts from epoch 1.
