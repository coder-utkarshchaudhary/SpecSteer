# IITD HPC — Ablation Grid Launcher Wiki

**Audience:** the lab junior who will run the 28-run ablation grid on the IITD HPC.  
**You need to touch exactly one script:** `bash scripts/hpc_launch.sh`.  
Everything else is automated.

> ⚠️ Anything marked **`FILL ME`** in this doc must be filled in *before* you run
> the launcher. Each `FILL ME` includes a one-line "how to obtain" hint. If any
> hint says "call Utkarsh" — do that, don't guess.

---

## Step 0 — Connect to the lab WiFi

**This is not optional.** The HPC is only reachable from the lab network.

- SSID: **`mlr lab 5g`**
- If you cannot see this SSID, you're not in the lab; go there.
- If you see it but can't connect, the WiFi password is on the whiteboard.

Confirm you're on the right network before continuing.

Copy the hpc_config.env.example into hpc_config.env and populate the fields. (SEE STEP 3)
You also need to set the LAB_DATA_ROOT in `scripts/hpc_config.env`.

---

## Step 1 — One-time setup on the lab machine

You only do this once. Skip to Step 2 if a previous junior already did it.

```bash
# System packages (Debian/Ubuntu):
sudo apt update
sudo apt install -y rsync openssh-client autossh tmux python3 python3-venv python3-pip
```

### 1a — Log in to Weights & Biases on the lab machine

The HPC job itself will run offline; the lab machine syncs the runs to the
wandb server after the job finishes.

```bash
wandb login
# Paste the API key when prompted.
```

**`WANDB_API_KEY`** — **FILL ME** — *ask Utkarsh (private channel). Never
commit or paste in the repo.*

### 1b — Passwordless SSH to the HPC
Ask Kavinder on how to connect to hpc. Or use what you learnt last time. If kavinder is not available or you can't figure out how to log in, then call Utkarsh.

```bash
# Copy your public key to the HPC (you'll type your HPC password once):
ssh-copy-id ${HPC_USER}@${HPC_HOST}

# Verify it works without a password:
ssh ${HPC_USER}@${HPC_HOST} 'echo ok'
```

**`HPC_USER`** — **FILL ME** — *your IITD LDAP username. Same as the SSH login
Utkarsh gives you.*

**`HPC_HOST`** — **FILL ME** — *IITD HPC login-node hostname. Usually
`padum.iitd.ac.in`. Confirm with Kavinder / Utkarsh / IITD HPC docs before using.*

---

## Step 2 — Verify processed data on the lab machine

The launcher rsyncs `data/processed/` to the HPC — the lab machine must have
the full set of preprocessed patches. **Do NOT run preprocessing** — the data
is already there.

```bash
ls "${LAB_DATA_ROOT:-/media/yashdeep/New Volume 21/UTKARSH_CHAUDHARY_prism/data/processed}"
# Expected output: AVIRIS  IIRS  M3  crims
```

If any of these subdirs are missing, **stop and call Utkarsh**.

---

## Step 3 — Fill `scripts/hpc_config.env`

```bash
cd /path/to/specsteer
cp scripts/hpc_config.env.example scripts/hpc_config.env
```

The launcher refuses to run until every `FILL_ME` is replaced. Here's the
catalog of what each field is and how to obtain it:

### HPC access

| Field | How to obtain |
|---|---|
| `HPC_USER` | Your IITD LDAP username. Ask Utkarsh if unsure. |
| `HPC_HOST` | Confirm with Utkarsh / IITD HPC docs. Often `padum.iitd.ac.in`. |
| `HPC_HOME` | Run: `ssh ${HPC_USER}@${HPC_HOST} 'echo $HOME'` — copy the path it prints. |
| `HPC_SCRATCH` | Usually `/scratch/${HPC_USER}` on IITD Padum. Verify: `ssh ${HPC_USER}@${HPC_HOST} 'ls -d /scratch/${HPC_USER}'`. |
| `HPC_PROJECT_DIR` | Defaults to `${HPC_HOME}/prism`. Leave as-is unless Utkarsh says otherwise. |

### PBS scheduler

| Field | How to obtain |
|---|---|
| `HPC_QUEUE` | List available queues: `ssh ${HPC_USER}@${HPC_HOST} 'qstat -q'`. Pick a GPU queue with A100 access. **Ask Utkarsh which queue is right.** |
| `HPC_WALLTIME` | Default `24:00:00`. If the queue's max is lower (check `qstat -q`), reduce it. |
| `HPC_SELECT` | Default is fine for a single A100 node with 8 CPUs, 64 GB memory. If the queue rejects it, ask Utkarsh — the exact syntax is IITD-specific. |
| `HPC_ARRAY_RANGE` | `1-28` for the full grid. For a smoke test set `1-1`. |
| `HPC_PROJECT_CODE` | Leave empty unless IITD assigned you one. |

### Lab-side tunnel

| Field | How to obtain |
|---|---|
| `LAB_TUNNEL_PORT` | Default `8765`. Change only if that port is already busy on the lab machine. |

### Telegram bot

| Field | How to obtain |
|---|---|
| `TELEGRAM_BOT_TOKEN` | **Ask Utkarsh.** He'll send it privately. |
| `TELEGRAM_CHAT_ID` | **Ask Utkarsh.** Same channel. |

### Weights & Biases

| Field | How to obtain |
|---|---|
| `WANDB_API_KEY` | **Ask Utkarsh.** The HPC job uses this only to write offline runs; sync happens on the lab machine (already logged in via Step 1a). |
| `WANDB_PROJECT` | Default `hsi-pi-vae`. Only change if Utkarsh says so. |
| `WANDB_ENTITY` | Leave blank unless Utkarsh gives you a team name. |

### Lab-side paths

| Field | How to obtain |
|---|---|
| `LAB_DATA_ROOT` | Default matches the lab drive path. Only change if the data lives elsewhere. |
| `LAB_REPO_ROOT` | Blank = auto-detect (uses this checkout). |

### Wheels

| Field | How to obtain |
|---|---|
| `PIP_PLATFORM`, `PIP_PYTHON_VERSION`, `PIP_ABI` | Defaults match Linux x86_64 + Python 3.11 on IITD HPC. If bootstrap complains about missing wheels, run `ssh ${HPC_USER}@${HPC_HOST} 'python3 --version'` and match the version. |

---

## Step 4 — Fire it up

```bash
cd /path/to/specsteer
bash scripts/hpc_launch.sh
```

That's it. The launcher does everything: builds wheels, rsyncs, bootstraps the
HPC venv, starts the lab-side relay, opens the reverse tunnel, kicks off the
HPC forwarder inside tmux, and submits the PBS array job.

**What to expect (in order):**

1. Sanity checks (WiFi, config, ssh key). Fails loudly if anything is off.
2. `pip download` of wheels — **5–15 min** the first time; skipped on later runs.
3. Rsync of the repo (fast), wheels (~1 GB, medium), and `data/processed/`
   (**this is the long one — 1–3 h over lab WiFi**).
4. Bootstrap on the HPC (offline pip install, `.env` creation).
5. Relay + tunnel + forwarder come up.
6. `qsub` — you get back a job ID.
7. Cheat-sheet printed with monitoring commands.

### Junior babysit rule

**Sit at the machine for the first 1.0–1.5 hours** while `data/processed/`
rsyncs. Watch the progress bar in the launcher log to confirm it's moving:

```bash
tail -f logs/hpc_launch_*.log
```
(This can be done by running the above command in a new terminal.)

Once you've seen the rsync progress advance steadily for ~30 min without
stalling, you may leave. The launcher runs in the foreground of your shell,
but the tunnel + forwarder + PBS job all keep running after it exits.

**Do not close your shell during the rsync phase.** After `qsub` completes,
you can close it — the tunnel is a background process with its PID in
`logs/tunnel.pid`.

---

## Step 5 — Monitoring

### On Telegram

You will get:

- **[START]** — once per run, with the resolved hyperparameters.
- **[HB]** — a heartbeat every 10 epochs with train/val loss, MSE, SAM, KLD.
- **[OK]** / **[FAIL]** / **[STOP]** — once per run, at the end.

### On the lab machine

```bash
# Launcher itself:
tail -f logs/hpc_launch_*.log

# Relay (Telegram sends from the tunnel land here):
tail -f logs/relay.log

# Tunnel status:
tail -f logs/tunnel.log

# Everything at a glance:
bash scripts/hpc_launch.sh --status
```

### On the HPC

```bash
# The PBS array queue:
ssh ${HPC_USER}@${HPC_HOST} 'qstat -t $(cat logs/hpc_jobid)'

# A specific array element's stdout+stderr:
ssh ${HPC_USER}@${HPC_HOST} 'tail -f ~/prism/logs/pbs_*_1.out'

# Attach to the forwarder tmux (detach with Ctrl-b, d):
ssh ${HPC_USER}@${HPC_HOST} 'tmux attach -t specsteer_forwarder'
```

---

## Step 6 — After the job completes

Telegram will send a `[OK]` message for each of the 28 runs. Once you have
28 finish messages (or the array status shows all elements done):

```bash
# Rsync checkpoints, wandb, logs back to the lab machine:
bash scripts/hpc_pull_results.sh

# Push offline wandb runs to the server:
cd /path/to/specsteer
wandb sync wandb/offline-run-*
```

**Verify you got 28 checkpoints:**

```bash
find model -name '*.pt' | wc -l
# Expect: 28
find model -name '*.pt' | sort
```

**Tell Utkarsh** — job is done, checkpoints pulled, wandb synced.

---

## Troubleshooting

| Symptom | What to do |
|---|---|
| `cannot ssh to ${HPC_USER}@${HPC_HOST}` | Passwordless key auth isn't set up. Redo Step 1b. |
| Rsync stalls, no progress for 10 min | Reconnect to `mlr lab 5g`. Re-run `bash scripts/hpc_launch.sh` — rsync is idempotent (`--partial`), it will resume. |
| `qsub` says "no matching queue" | Wrong `HPC_QUEUE`. Run `ssh ${HPC_USER}@${HPC_HOST} 'qstat -q'` and pick a valid GPU queue. Ask Utkarsh. |
| `qsub` says "resources not available" | The A100 nodes are busy. This will resolve on its own — PBS will schedule as slots free up. Nothing to do. |
| No Telegram messages arriving | Check `logs/relay.log` on the lab and `ssh ${HPC_USER}@${HPC_HOST} 'tail logs/forwarder.log'`. Common cause: tunnel died — run `bash scripts/hpc_launch.sh --status`; re-run `bash scripts/hpc_launch.sh` to restart it. |
| Forwarder tmux session died | `ssh ${HPC_USER}@${HPC_HOST} 'tmux kill-session -t specsteer_forwarder'` then re-run `bash scripts/hpc_launch.sh` — it will restart the forwarder. |
| Bootstrap says "wheels/ empty" | `pip download` on the lab produced 0 files. Usually a `PIP_PLATFORM` mismatch. Check `ssh ${HPC_USER}@${HPC_HOST} 'python3 --version'` and update `PIP_PYTHON_VERSION` / `PIP_ABI` in the config to match. |
| A single run says `[FAIL]` on Telegram | The Telegram message includes the last log lines and the exception. Screenshot it and send to Utkarsh; don't try to fix it yourself. The other 27 runs continue independently. |

---

## Emergency stop

If Utkarsh tells you to kill everything:

```bash
bash scripts/hpc_launch.sh --stop
```

That kills the tunnel, the local relay, the HPC forwarder tmux session, and
runs `qdel` on the PBS array. All 28 array elements terminate.

---

## Summary — what you actually do

1. Connect to `mlr lab 5g` WiFi.
2. First time only: install packages, `wandb login`, `ssh-copy-id`.
3. `cp scripts/hpc_config.env.example scripts/hpc_config.env` and fill every
   `FILL_ME`. Ask Utkarsh for the values marked "ask Utkarsh".
4. `bash scripts/hpc_launch.sh` — sit at the machine for 1.0–1.5 h until
   the rsync bar is clearly progressing, then you can leave.
5. Wait for the Telegram finish messages.
6. `bash scripts/hpc_pull_results.sh` and `wandb sync wandb/offline-run-*`.
7. Tell Utkarsh.

That is the entire junior workflow. If something isn't covered above — don't
improvise, ping Utkarsh.
