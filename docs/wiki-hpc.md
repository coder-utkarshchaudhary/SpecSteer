# Wiki for running the code successfully — IITD HPC (Padum)

> This wiki targets the **IITD Padum cluster** (`hpc.iitd.ac.in`) via PBS Pro.
> All long-running work goes through `qsub`. Login nodes are for
> compilation, submitting jobs, and inspecting files only.
> Quoted directly from the HPC docs:
> _"Please do not run programs directly on login nodes"_ and
> _"$SCRATCH is NOT backed up! Please download all your data!"_

**Compute nodes have no outbound internet.** That's the whole reason this
workflow exists: everything internet-dependent (pip, Google Drive, wandb)
happens on the lab Mac first. You end up on HPC with a self-contained tree
that behaves like an offline workstation.

Follow the steps in order. **If any step fails, stop and text me a screenshot
before improvising.**

---

## 1. Access & sanity check

1. **SSH in.** From inside the IITD network:
   ```bash
   ssh <kerberos-id>@hpc.iitd.ac.in
   ```
   From outside IITD, connect to the IITD VPN first (see the CSC website), then
   the same command. If VPN doesn't work, email `hpchelp@iitd.ac.in` via your
   supervisor.

2. **Note your project code.** Every `qsub` needs `-P <dept>`. Run:
   ```bash
   echo $HOME
   ```
   The segment right after `/home/` is your project code (e.g. `/home/cc/...`
   → project code is `cc`). **Write this down** — `install_on_hpc.sh` needs it.

3. **Sanity check the environment.** Run these on the login node and
   send me the outputs:
   ```bash
   whoami                    # your username
   echo "PROJ=$(echo $HOME | cut -d/ -f3)"    # your project code
   quota                     # home quota (should be ~100 GB)
   df -h $SCRATCH            # scratch usage (25 TB total)
   qstat -Q                  # list available queues
   ```
   You should see `standard` in the queue list. Home is 100 GB backed up,
   scratch is 25 TB **not** backed up.

4. **Do NOT run `nvidia-smi` or training on the login node.** Login nodes
   have no GPU visible; GPU checks happen inside a PBS job (see step 5).

---

## 2. Build the bundle on the lab Mac

Everything below happens on the Mac, **before** you touch the HPC again.

### 2.1. Drop the datasets you have on hand into `data/processed/`

The HPC bundle ships the processed data — no preprocessing on HPC. Two of
the four datasets we already have; two come from Drive.

From the external hard drive, copy over:

```bash
cd ~/Documents/personal/specsteer      # repo root
mkdir -p data/processed
cp -R /Volumes/<drive>/prism-data-processed/M3    data/processed/
cp -R /Volumes/<drive>/prism-data-processed/crims data/processed/
```

Note the mixed casing — `M3`, `crims` (lowercase) are load-bearing (see
`utils/config.py::DATASETS`).

### 2.2. Run the builder

```bash
bash scripts/build_hpc_bundle.sh \
     --drive-url "https://drive.google.com/drive/folders/1QjwlQRSCgLFKT4f3SHYTOyFSAKXiIAlZ"
```

What this does:
1. `gdown`s IIRS + AVIRIS processed folders from Drive into
   `data/processed/` (skips any that are already there).
2. `pip download`s all wheels for **Python 3.10 / linux_x86_64 / CUDA 12.1**
   into `build/hpc_bundle/wheels/` (~2 GB).
3. Hardlink-copies `data/processed/` into `build/hpc_bundle/data/processed/`.
4. Tars the repo (excluding `.git`, `.venv`, `data/`, `model/`, etc.) into
   `build/hpc_bundle/code.tar.gz`.
5. Writes `README.txt` with the exact scp commands.

Skip the gdown step if all four folders are already local:

```bash
bash scripts/build_hpc_bundle.sh --skip-gdown
```

### 2.3. Ship the bundle to HPC

Two independent transfers — the first is small (~2 GB), the second is
big (~35 GB) and worth `rsync`ing so you get resume-on-drop.

```bash
cd build/hpc_bundle

# code + wheels → $HOME on Padum
scp   code.tar.gz install.sh <user>@hpc.iitd.ac.in:~/
rsync -avP wheels/           <user>@hpc.iitd.ac.in:~/prism-wheels/

# processed data → $SCRATCH
rsync -avP data/processed/ \
      <user>@hpc.iitd.ac.in:/scratch/<proj>/<user>/prism-data/processed/
```

`(<proj>` is your project code; `<user>` is your kerberos id. Confirm the
scratch path with `ssh <user>@hpc.iitd.ac.in 'echo $SCRATCH'` first.)

---

## 3. Install on the HPC login node

```bash
ssh <user>@hpc.iitd.ac.in
cd ~
tar -xzf code.tar.gz                       # → creates ./specsteer/ (or ./prism/)
mv install.sh specsteer/scripts/           # only if you keep them side by side
cd specsteer

PROJECT_CODE=cc \
IITD_EMAIL=<you>@iitd.ac.in \
WHEELS_DIR=$HOME/prism-wheels \
bash scripts/install_on_hpc.sh
```

What this does:
1. Fills `<REPLACE_ME>` in `scripts/hpc_train.pbs` with your project code +
   email.
2. Creates `.venv/` and installs from `~/prism-wheels/` with `--no-index`
   (no PyPI access needed).
3. Creates `$SCRATCH/prism-data/{original,processed}` and symlinks
   `data/original`, `data/processed` under the repo. If you already
   `rsync`'d directly to `$SCRATCH`, the symlink just picks it up.
4. Prints the next commands to run.

---

## 4. Sanity check v2

Still on the login node, venv activated:

```bash
cd $HOME/specsteer
source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
python3 utils/check-model-params.py
```

Send me the output on WhatsApp and call me.

Optional: log wandb in **offline mode** (compute nodes have no outbound
internet, but the credentials file itself just needs to exist for the
offline runs to write correctly):

```bash
wandb login --relogin        # paste your API key from wandb.ai/authorize
```

---

## 5. Smoke test on a GPU compute node (interactive)

Before submitting the 168 h monster, prove end-to-end works with a 1-hour
interactive GPU session:

```bash
qsub -P <PROJECT_CODE> -I -l select=1:ncpus=4:ngpus=1 -l walltime=1:00:00
```

Once you land on the compute node:

```bash
cd $HOME/specsteer
source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
export WANDB_MODE=offline

nvidia-smi                              # confirm GPU (A100 or V100)
free -h                                 # host RAM

# 1-epoch training smoke on the flagship model
python train/train.py --model vae-our --dataset IIRS --loss physics --epochs 1
ls -lh model/IIRS/vae-our.pt

# logging + inference smoke
python inference/inference.py --model vae-our --dataset IIRS --loss physics
grep -E "MSE|SAM|PSNR|SSIM" logs/inference_vae-our_IIRS_*.log
```

**Delete the ckpt after the smoke test** so the real job doesn't skip it:
```bash
rm model/IIRS/vae-our.pt
```
Then `exit` to leave the interactive session.

---

## 6. Submit the full 28-run grid

```bash
cd $HOME/specsteer
qsub scripts/hpc_train.pbs
qstat -u $USER                     # confirm state Q or R
qstat -T <jobid>                   # estimated start time (recomputed every 6h)
```

Monitor:
```bash
tail -f logs/hpc/train28.out
```

**Failure & resume behaviour** — the grid is idempotent:
- Each of the 28 runs is wrapped so **one failure does not kill the other 27**.
- Runs whose checkpoint already exists at `model/<DATASET>/<name>.pt` are
  **skipped** on re-submission.
- If walltime expires or the node crashes, **just `qsub scripts/hpc_train.pbs`
  again** — it'll pick up where it left off.

At the end of `logs/hpc/train28.out` you'll see:

```
==============================================
 Ablation grid complete.
  skipped (ckpt exists) : N
  failed                : M
  failed runs:
    - <model>|<dataset>|<loss>|rc=<code>
==============================================
```

If `failed > 0`, forward those lines to me — don't send the whole log.

The 28 checkpoints, when all present, are:

```
model/IIRS/{vae-our, vae-standard_standard, vae-standard_physics,
            vae-3d-spatio-spectral_standard, vae-3d-spatio-spectral_physics,
            vae-1d-pixelwise_standard, vae-1d-pixelwise_physics}.pt
model/M3/{same 7}.pt
model/AVIRIS/{same 7}.pt
model/CRIMS/{same 7}.pt
```

Total: **28**. Verify with:
```bash
find model -name "*.pt" | wc -l
```

---

## 7. After the grid finishes

1. **Sync wandb** from the login node:
   ```bash
   cd $HOME/specsteer
   source .venv/bin/activate
   wandb sync wandb/offline-run-*
   ```
2. **Snapshot the checkpoints** so future re-submits don't overwrite them:
   ```bash
   cp -r model/ $HOME/prism-checkpoints-$(date +%F)/
   ```
3. **Downstream experiments** (needs GPU; submit as a fresh short PBS job):
   ```bash
   qsub -P <PROJECT_CODE> -I -l select=1:ncpus=4:ngpus=1 -l walltime=4:00:00
   # on the compute node:
   cd $HOME/specsteer && source .venv/bin/activate
   export PYTHONPATH=$PWD:$PYTHONPATH
   python inference/downstream.py --dataset IIRS   --save-plots
   python inference/downstream.py --dataset M3     --save-plots
   python inference/downstream.py --dataset AVIRIS --save-plots
   python inference/downstream.py --dataset CRIMS  --save-plots
   ```
4. **Pull results back to the Mac** so they're safe from the scratch purge:
   ```bash
   # from the Mac
   rsync -avP <user>@hpc.iitd.ac.in:~/specsteer/model/           ./model/
   rsync -avP <user>@hpc.iitd.ac.in:~/specsteer/visualisations/  ./visualisations/
   rsync -avP <user>@hpc.iitd.ac.in:~/specsteer/wandb/           ./wandb/
   ```

Text me when all 28 checkpoints are in `model/` and downstream plots are
generated. Call me if anything above breaks.
