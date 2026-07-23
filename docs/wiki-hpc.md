# Wiki for running the code successfully — IITD HPC (Padum)

> This wiki targets the **IITD Padum cluster** (`hpc.iitd.ac.in`) via PBS Pro.
> All long-running work goes through `qsub`. Login nodes are for
> compilation, submitting jobs, and inspecting files only.
> Quoted directly from the HPC docs:
> _"Please do not run programs directly on login nodes"_ and
> _"$SCRATCH is NOT backed up! Please download all your data!"_

Here are the steps I need you to follow, in order. **If any step fails, stop
and text me a screenshot before improvising.**

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
   → project code is `cc`). **Write this down** — you will paste it into the
   two `.pbs` files.

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
   have no GPU visible; GPU checks happen inside a PBS job (see step 6).

---

## 2. Repo clone

Code is small and belongs on backed-up `$HOME`:

```bash
cd $HOME
git clone https://github.com/coder-utkarshchaudhary/SpecSteer.git prism
cd prism
git checkout master
```

Confirm you're on master with `git log -1 --oneline`.

---

## 3. Env setup

Env creation is allowed on the login node (takes < 2 minutes):

```bash
cd $HOME/prism
python3 -m venv .venv
source .venv/bin/activate
which python3            # must show $HOME/prism/.venv/bin/python3
pip install -r requirements.txt
pip install gdown wandb
```

If `python3 -m venv` fails, try `python -m venv` and retry. If both fail, call me.

Once installed, log wandb in **offline mode** for now (compute nodes have no
outbound internet in most partitions):

```bash
wandb login --relogin        # paste your API key from wandb.ai/authorize
```

---

## 4. Data download → `$SCRATCH`

The dataset is **~217 GB** across 4 dataset folders (IIRS, M3, AVIRIS, CRIMS).
It cannot fit under `$HOME` (100 GB quota) and must live on `$SCRATCH`
(25 TB, not backed up).

### 4.1. Prepare the scratch layout

```bash
mkdir -p $SCRATCH/prism-data/original
cd $HOME/prism
mkdir -p data
ln -sfn $SCRATCH/prism-data/original data/original
readlink data/original           # verify: prints $SCRATCH/prism-data/original
```

### 4.2. Edit the two PBS files with your project code

Open both files and replace `<REPLACE_ME>` with your project code and
`REPLACE_ME@iitd.ac.in` with your IITD email:

```bash
# open in your favourite editor:
nano scripts/hpc_download.pbs
nano scripts/hpc_train.pbs
```

Fields to change in each:
- `#PBS -P <REPLACE_ME>` → `#PBS -P cc` (or your actual code)
- `#PBS -M REPLACE_ME@iitd.ac.in` → your address

### 4.3. Submit the download job (preferred path: `gdown`)

```bash
mkdir -p logs/hpc
qsub scripts/hpc_download.pbs
qstat -u $USER                  # confirm the job appears in state Q or R
```

You'll get an email when it starts and finishes (~1-3 h realistic; walltime
cap is 24 h). While waiting, tail the logs:

```bash
tail -f logs/hpc/download.out
```

### 4.4. Fallback if `gdown` hits the Google 24 h quota

`gdown --continue --remaining-ok` will resume where it left off on a fresh
submission, so **first try re-submitting the same job** the next day.

If that still fails, use rclone. One-time setup on the **login node**:

```bash
pip install --user rclone      # or: module load rclone
rclone config
# n) new remote
# name> gdrive
# type> drive
# client_id / secret> leave blank (or set your own for better quota)
# scope> 2  (read-only)
# When it prints an authorize URL, open it in your laptop browser,
# log in with the Google account that has access to the folder,
# copy the auth token back to the terminal.
```

Then submit the download using rclone instead:

```bash
rclone copy 'gdrive:{path to prism-data on drive}' \
       $SCRATCH/prism-data/original \
       --drive-shared-with-me --progress --transfers 8
```

Wrap that inside an interactive PBS session so the login node doesn't kill it:

```bash
qsub -P <REPLACE_ME> -I -l select=1:ncpus=4 -l walltime=12:00:00
# ...once you land on the compute node, run the rclone command above
```

### 4.5. Verify the download

```bash
du -sh $SCRATCH/prism-data/original           # ~ 217 GB
ls $SCRATCH/prism-data/original               # 4 folders
```

Expect exactly these four:

```
data/original - IIRS
data/original - m3
data/original - AVIRIS
data/original - CRIMS
```

If any are missing, re-submit `scripts/hpc_download.pbs` (safe — resumes).

---

## 5. Sanity check v2

Still on the login node, venv activated:

```bash
cd $HOME/prism
source .venv/bin/activate
python3 utils/check-model-params.py
```

Send me the output on WhatsApp and call me.

---

## 6. Smoke test on a GPU compute node (interactive)

Before submitting the 168 h monster, prove end-to-end works with a 1-hour
interactive GPU session:

```bash
qsub -P <REPLACE_ME> -I -l select=1:ncpus=4:ngpus=1 -l walltime=1:00:00
```

Once you land on the compute node, verify GPU visibility and run the smoke tests:

```bash
cd $HOME/prism
source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
export WANDB_MODE=offline

nvidia-smi                              # confirm GPU (A100 or V100)
free -h                                 # host RAM

# preprocess just one folder
bash scripts/preprocess.sh --dataset iirs --limit 1
ls $SCRATCH/prism-data/processed/IIRS/train | head

# 1-epoch training smoke on each real model
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

## 7. Submit the full 28-run grid

```bash
cd $HOME/prism
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

## 8. After the grid finishes

1. **Sync wandb** from the login node (compute nodes usually have no outbound
   internet):
   ```bash
   cd $HOME/prism
   source .venv/bin/activate
   wandb sync wandb/offline-run-*
   ```
2. **Snapshot the checkpoints** so future re-submits don't overwrite them:
   ```bash
   cp -r model/ $HOME/prism-checkpoints-$(date +%F)/
   ```
3. **Downstream experiments** (needs GPU; submit as a fresh short PBS job):
   ```bash
   qsub -P <REPLACE_ME> -I -l select=1:ncpus=4:ngpus=1 -l walltime=4:00:00
   # on the compute node:
   cd $HOME/prism && source .venv/bin/activate
   export PYTHONPATH=$PWD:$PYTHONPATH
   python inference/downstream.py --dataset IIRS --save-plots
   python inference/downstream.py --dataset M3 --save-plots
   python inference/downstream.py --dataset AVIRIS --save-plots
   python inference/downstream.py --dataset CRIMS --save-plots
   ```

Text me when all 28 checkpoints are in `model/` and downstream plots are
generated. Call me if anything above breaks.
