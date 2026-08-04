# Running the Ablation Grid on IITD HPC

This document walks you through everything from "I just sat down at the lab machine" to "checkpoints are back and Utkarsh is notified." Read it top-to-bottom like a recipe — every step depends on the ones before it.

You will touch exactly one script: **`bash scripts/hpc_launch.sh`**. Everything else is automated. If something breaks and this doc doesn't cover it, don't improvise — call Utkarsh.

---

## Before you begin
Sit on system 9 (password is mlr@123#; confirm kar liyo idr properly). Connect to the **`mlr lab 5g`** WiFi. Other wifi networks are slow and will take a lot of time in rsync. 

---

## Step 1 — Pull the latest code

Open a terminal on the lab machine and navigate to the repo. Mujhe exactly nahi pata ye kaha hai but it is most likely here `cd media/mlr/New Volume 21/...`; ek baar call me. I'll tell you where it is.

```bash
cd <path-to-repo-given-by-utkarsh>              # or wherever the repo lives
git checkout -b hpc                             # switch to the hpc branch
git pull origin hpc                             # get the latest changes
```

If git asks you to stash local changes, do:

```bash
git stash
git pull origin hpc
git stash pop               # brings your changes back on top
```

**If this step fails** — you probably have uncommitted edits that conflict. Call Utkarsh rather than guessing at a merge resolution.

---

## Step 2 — One-time machine setup

Skip this step entirely if a previous person already did it (the test in Step 3 will confirm). You only run these commands once per lab machine.
Install the system tools the launcher depends on:

```bash
sudo apt update
sudo apt install -y rsync openssh-client autossh tmux python3 python3-venv python3-pip
```

Set up passwordless SSH to the HPC so the launcher can run non-interactively. Ask Kavinder how to connect to the HPC. If Kavinder is unavailable or you can't figure it out, call Utkarsh.

Once you know your HPC username and hostname, run:

```bash
ssh-copy-id YOUR_HPC_USER@THE_HPC_HOST
```

It will ask for your HPC password once. After that, verify:

```bash
ssh YOUR_HPC_USER@THE_HPC_HOST 'echo ok'
```

If it prints `ok` without asking for a password, you're set. If it still asks for a password, something went wrong — ask Kavinder or Utkarsh.
Finally, log in to Weights & Biases on the lab machine (Utkarsh will give you the API key privately) using a new terminal window. Ensure the .prism-venv is activates using conda (`conda activate .prism-venv`)

```bash
pip install wandb
wandb login
# Paste the API key when prompted.
```

---

## Step 3 — Fill the configuration file

The launcher reads all its settings from a single file. Copy the template by running the following commands in the terminal:

```bash
cp scripts/hpc_config.env.example scripts/hpc_config.env
```

Open `scripts/hpc_config.env` in VS Code. Every line that says `FILL_ME` needs to be replaced with a real value. Here is where to get each one:

**HPC credentials (ask Kavinder or Utkarsh):**

- `HPC_USER` — your IITD login username.
- `HPC_HOST` — the HPC login node (typically `padum.iitd.ac.in`). Confirm with Kavinder.
- `HPC_HOME` — run `ssh YOUR_USER@THE_HOST 'echo $HOME'` and paste what it prints.
- `HPC_SCRATCH` — run `ssh YOUR_USER@THE_HOST 'ls -d /scratch/YOUR_USER'` and paste the path.

**PBS queue (ask Utkarsh):**

- `HPC_QUEUE` — which GPU queue to submit to. If you want to check yourself: `ssh YOUR_USER@THE_HOST 'qstat -q'` lists them. Pick the one with A100 access. When in doubt, ask Utkarsh.

**Telegram bot (ask Utkarsh — he will DM you):**

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

**Wandb (ask Utkarsh):**

- `WANDB_API_KEY`

**Data path:**

- `LAB_DATA_ROOT` — where the processed patches live on this machine. The default is `/media/yashdeep/New Volume 21/UTKARSH_CHAUDHARY_prism/data/processed`. If the data is elsewhere, update it.

Everything else has sensible defaults. Don't change them unless Utkarsh tells you to.

**How to verify this step worked:** the launcher will check every field and refuse to start if anything is still `FILL_ME` or empty. So if Step 4 starts running, your config is correct.

---

## Step 4 — Verify the data is present

The launcher will rsync `data/processed/` to the HPC. Confirm it exists locally:

```bash
ls "${LAB_DATA_ROOT}"
```

You should see subdirectories: `IIRS`, `M3`, `AVIRIS`, `crims`. If any are missing, **stop and call Utkarsh** — do not proceed without the full dataset.

---

## Step 5 — Launch
In a new terminal window run the following command from repo root:
```bash
bash scripts/hpc_launch.sh
```

That's it. The script does everything automatically:

1. Checks WiFi, config, SSH access, local data.
2. Downloads pip wheels for the HPC platform (5–15 min first time; instant on subsequent runs).
3. Rsyncs the repo, wheels, and processed data to the HPC (**this is the slow step — 1 to 3 hours over lab WiFi**).
4. Runs the HPC bootstrap (creates the Python environment from the wheels, offline).
5. Starts the Telegram relay on the lab machine.
6. Opens a reverse SSH tunnel so HPC notifications reach the lab. (This might ask for the lab system password, I am not sure yet. If it does just ask any faculty.)
7. Starts the login-node message forwarder inside tmux on the HPC.
8. Submits a **smoke run** — one slot, 5 epochs — to verify the environment works.
9. Starts a background watcher that waits for the smoke to finish.

**What happens after the smoke:**

- If the smoke passes: the watcher sleeps 10 minutes (giving you a window to stop if something looked wrong), then automatically submits the full 28-run grid. You will get a Telegram message: `[LAUNCHED] Full ablation grid submitted`.
- If the smoke fails: the watcher pulls the logs, sends the tail to Telegram, and stops. The full grid is NOT submitted. Fix the issue (or call Utkarsh), then re-run `bash scripts/hpc_launch.sh`.

**How long should you sit here?**

Stay at the machine for 1.0–1.5 hours while the data rsyncs. Open a second terminal and watch:

```bash
tail -f logs/hpc_launch_*.log
```

As long as you see the rsync progress bar moving, things are fine. Once you've confirmed it's been moving steadily for ~30 minutes without stalling, you can leave. The script runs in the foreground, but the tunnel, relay, forwarder, and watcher are all background processes that survive you closing the terminal.

**Do not close the terminal while rsync is still running.** After the script prints "launch complete," you can close it safely.

**If rsync stalls** (no progress for 10+ minutes): your WiFi probably dropped. Reconnect to `mlr lab 5g` and re-run `bash scripts/hpc_launch.sh`. Rsync resumes where it left off (`--partial`), so you don't lose progress.

---

## Step 6 — Monitor

Utkarsh will receive Telegram messages throughout the run:

- **[START]** — one per run, showing the hyperparameters.
- **[HB]** — every 10 epochs, with current loss / MSE / SAM / KLD metrics, wall time, and ETA.
- **[OK]** / **[FAIL]** / **[STOP]** — when a run finishes.

If you want to check status from the lab machine:

```bash
bash scripts/hpc_launch.sh --status
```

This shows whether the tunnel, relay, watcher, and PBS job are alive.

If you want to watch logs on the HPC:

```bash
# PBS array queue:
ssh YOUR_USER@THE_HOST 'qstat -t $(cat ~/prism/logs/hpc_jobid)'

# A specific run's log:
ssh YOUR_USER@THE_HOST 'tail -f ~/prism/logs/train_*.log'

# The forwarder (detach with Ctrl-b then d):
ssh YOUR_USER@THE_HOST 'tmux attach -t specsteer_forwarder'
```

---

## Step 7 — Pull results back

When all 28 runs are done (you'll get 28 `[OK]` messages on Telegram, or check `qstat` shows them all finished):

```bash
cd ~/specsteer
bash scripts/hpc_pull_results.sh
```

This rsyncs the checkpoints, wandb offline runs, and logs from the HPC to the lab machine.

Then push the wandb runs to the server:

```bash
wandb sync wandb/offline-run-*
```

Verify you got 28 checkpoints:

```bash
find model -name '*.pt' | wc -l
# Should print: 28
```

**Tell Utkarsh** — job done, checkpoints pulled, wandb synced.

---

## Emergency stop

If something is wrong and you need to kill everything:

```bash
bash scripts/hpc_launch.sh --stop
```

This kills the local relay, tunnel, and watcher; kills the HPC forwarder tmux session; and `qdel`s both the smoke and full PBS jobs. All array elements terminate.

---

## Troubleshooting

**"cannot ssh to ..."** — Passwordless key auth isn't set up. Go back to Step 2.

**Rsync stalls** — Reconnect to `mlr lab 5g`. Re-run `bash scripts/hpc_launch.sh` — it resumes.

**qsub says "no matching queue"** — Wrong `HPC_QUEUE` in the config. Run `ssh YOUR_USER@HOST 'qstat -q'` and pick a valid GPU queue. Ask Utkarsh.

**qsub says "resources not available"** — A100 nodes are busy. PBS will schedule when slots free. Nothing to do.

**"reverse tunnel failed to start"** — Two causes. Either your lab→HPC SSH key has a passphrase (the backgrounded tunnel can't type it), or the HPC login node blocks port-forwarding. For the first: make a passphraseless key, or run `eval $(ssh-agent); ssh-add ~/.ssh/id_ed25519` once, then re-launch. For the second, call Utkarsh — it needs a config change on his side. Note: the tunnel logs in *to the HPC* using your HPC key — it never asks for the lab machine's password.

**"tunnel is up but the HPC could not reach the relay"** — The launcher warns about this but keeps going. Telegram messages are not lost — they queue on the HPC in `logs/notify_queue.jsonl` and flush automatically once the tunnel works. If messages never arrive, the login node is blocking forwarding; tell Utkarsh.

**No Telegram messages arriving** — Check `logs/relay.log` and the tunnel: `bash scripts/hpc_launch.sh --status`. If the tunnel died, just re-run the launcher — it restarts it.

**Smoke failed** — Read the Telegram failure message. Screenshot it and send to Utkarsh. Do not attempt to fix code yourself.

**"wheels/ empty" during bootstrap** — The `PIP_PYTHON_VERSION` in the config doesn't match the HPC's python. Run `ssh YOUR_USER@HOST 'python3 --version'` and update `PIP_PYTHON_VERSION` and `PIP_ABI` to match.

**A single run failed but others passed** — That's expected for edge cases. The other 27 runs continue independently. Tell Utkarsh about the failure; he'll decide whether to re-run it.

---

## Summary

1. Connect to `mlr lab 5g`.
2. `git checkout hpc && git pull origin hpc`.
3. One-time only: install packages, SSH keys, `wandb login`.
4. Fill `scripts/hpc_config.env` (ask Utkarsh / Kavinder for every `FILL_ME`).
5. Confirm data exists at `LAB_DATA_ROOT`.
6. `bash scripts/hpc_launch.sh` — sit for 1–1.5 h during rsync.
7. Wait for Telegram messages.
8. `bash scripts/hpc_pull_results.sh` + `wandb sync wandb/offline-run-*`.
9. Tell Utkarsh.

If it's not in this list, don't do it. If something fails, call Utkarsh.
