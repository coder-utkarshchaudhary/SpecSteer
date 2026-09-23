# Running the Ablation Grid on HPC

This document walks you through everything from "I just sat down at the lab machine" to "checkpoints are back and results are synced." Read it top-to-bottom — every step depends on the ones before it.

You will touch exactly two scripts: **`bash scripts/hpc_preflight.sh`** (read-only checks, run first) and **`bash scripts/hpc_launch.sh`** (does the actual work). Everything else is automated. If something breaks and this doc does not cover it, do not improvise — consult your PI.

---

## The two-node topology (read this once, it explains everything below)

The HPC cluster has **two separate machines with two separate filesystems**:

- The **login node** — what `HPC_HOST` in the config points at. Reached directly from the
  lab as `ssh ${HPC_USER}@${HPC_HOST}`. This machine does **not** have `qsub` — it is purely a
  staging/jump host.
- The **compute node** — reached by first `ssh`-ing into the login node, and from *there*
  running `ssh hpc` (or whatever `HPC_INNER_HOST` is set to in the config). This is where jobs
  actually run and where `qsub`/`qstat`/`qdel` live.

Everything in `scripts/hpc_launch.sh` therefore happens in two hops: repo/venv/data get pushed
lab → login → compute; results come back compute → login → lab; Telegram notifications go
lab → login → compute (a tunnel) and compute → login → lab (the messages themselves). You never
have to manage this by hand — `scripts/hpc_common.sh` wraps both hops for every script — but
when something looks stuck, knowing which hop it is stuck on is the first diagnostic question.

---

## Step 1 — Pull the latest code

Open a terminal on the lab machine and navigate to the repo:

```bash
cd <path-to-repo>
git checkout hpc
git pull origin hpc
```

If git asks you to stash local changes, do:

```bash
git stash
git pull origin hpc
git stash pop
```

If this step fails, you probably have uncommitted edits that conflict. Resolve the conflict or ask your PI.

---

## Step 2 — One-time machine setup

Skip this step entirely if a previous person already did it (the test in Step 3 will confirm). You only run these commands once per lab machine.

Install the system tools the launcher depends on:

```bash
sudo apt update
sudo apt install -y rsync openssh-client autossh tmux python3 python3-venv python3-pip
```

Set up passwordless SSH to the HPC so the launcher can run non-interactively. Ask your HPC administrator for your HPC username and the hostname. Once you have them:

```bash
ssh-copy-id YOUR_HPC_USER@THE_HPC_HOST
```

It will ask for your HPC password once. After that, verify:

```bash
ssh YOUR_HPC_USER@THE_HPC_HOST 'echo ok'
```

If it prints `ok` without asking for a password, you are set. If it still asks for a password, something went wrong — ask your HPC administrator.

**Also verify the second hop** — once logged into the login node, `ssh hpc` (or whatever
`HPC_INNER_HOST` is set to) must work *without a password* too, since that is how the launcher
reaches the compute node where `qsub` actually lives:

```bash
ssh YOUR_HPC_USER@THE_HPC_HOST 'ssh hpc echo ok'
```

If that does not print `ok`, ask your HPC administrator — the login node's own SSH key needs to
be authorized on the compute node, which is a one-time setup on the HPC side.

Finally, log in to Weights & Biases. This repo's Python environment lives in `.venv/` (a plain venv, not conda) — activate it first:

```bash
source .venv/bin/activate
pip install wandb   # only if wandb isn't already in .venv
wandb login
# Paste the API key when prompted.
```

---

## Step 3 — Fill the configuration file

Copy the template:

```bash
cp scripts/hpc_config.env.example scripts/hpc_config.env
```

Open `scripts/hpc_config.env`. Every line that says `FILL_ME` needs a real value:

**HPC credentials:**

- `HPC_USER` — your HPC login username.
- `HPC_HOST` — the HPC login node hostname (e.g. `login.cluster.example.edu`).
- `HPC_HOME` — run `ssh YOUR_USER@THE_HOST 'echo $HOME'` and paste what it prints.
- `HPC_INNER_HOST` — what you type after `ssh` once already on the login node to reach the
  compute node (default `hpc`). Verify with
  `ssh YOUR_USER@THE_HOST 'ssh HPC_INNER_HOST echo ok'`.

**PBS queue:**

- `HPC_QUEUE` — which GPU queue to submit to. Check available queues: `ssh -t YOUR_USER@THE_HOST 'qstat -q'`. Pick the one with A100 access.

**Telegram bot (optional — for training notifications):**

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

**Weights & Biases:**

- `WANDB_API_KEY`

**Data path:**

- `LAB_DATA_ROOT` — where the processed patches live on this machine (e.g. `/path/to/data/processed`).

Everything else has sensible defaults.

**Verification:** the launcher checks every field and refuses to start if anything is still `FILL_ME` or empty.

---

## Step 4 — Verify the data is present

```bash
ls "${LAB_DATA_ROOT}"
```

You should see subdirectories: `IIRS`, `M3`, `AVIRIS`, `CRIMS` (exact case — a lowercase name will silently fail the on-HPC data check later). If any are missing, do not proceed without the full dataset.

---

## Step 5 — Preflight (read-only, always run this first)

```bash
bash scripts/hpc_preflight.sh
```

This checks, without changing anything: that you can reach the login node, that the login node
can reach the compute node, that both nodes have the tools the launcher needs, and — if this
repo's `.venv/` has already been pushed to the HPC — that it actually imports `torch` and
`wandb` there. Fix anything it reports as `FAIL` before continuing; `WARN` lines are fine to
proceed past (they are usually "not pushed yet," which is expected on a first run).

---

## Step 6 — Launch

```bash
bash scripts/hpc_launch.sh
```

That is it. The script does everything automatically:

1. Checks network connectivity, config, SSH access to both nodes, local data.
2. Skips the pip-wheel build entirely (this repo ships a prebuilt `.venv/`; see
   `USE_SHIPPED_VENV` in the config).
3. Rsyncs the repo, `.venv/`, and processed data from the **lab to the login node** — each of
   the three is pushed independently and only if missing or broken on the login node
   (**this is the slow step when it runs — 1 to 3 hours over a typical network for the data**).
4. Rsyncs the same three things again from the **login node to the compute node**.
5. Runs the HPC bootstrap **on the compute node** (verifies the shipped `.venv` works; only
   falls back to an offline pip install if you set `USE_SHIPPED_VENV=0`).
6. Starts the Telegram relay on the lab machine, a reverse tunnel chained lab → login → compute,
   a message forwarder in tmux **on the compute node**, and a results collector in tmux **on the
   login node**.
7. Submits a **smoke run** — one slot, 5 epochs — to verify the environment works.
8. Starts a background watcher that waits for the smoke to finish.

**What happens after the smoke:**

- If the smoke passes: the watcher sleeps 10 minutes (giving you a window to stop if something looked wrong), then automatically submits the full grid. You will get a Telegram message: `[LAUNCHED] Full ablation grid submitted`.
- If the smoke fails: the watcher pulls the logs, sends the tail to Telegram, and stops. The full grid is NOT submitted. Fix the issue, then re-run `bash scripts/hpc_launch.sh`.

**How long to wait:** stay at the machine while the data rsyncs. Open a second terminal and watch:

```bash
tail -f logs/hpc_launch_*.log
```

As long as you see rsync progress moving, things are fine. The script runs in the foreground, but the tunnel, relay, forwarder, and watcher are all background processes that survive you closing the terminal once the launch step is done.

**Do not close the terminal while rsync is still running.** After the script prints "launch complete," you can close it safely.

---

## Step 7 — Monitor

Telegram messages arrive throughout the run:

- **[START]** — one per run, showing the hyperparameters.
- **[HB]** — every 10 epochs, with current loss / MSE / SAM / KLD metrics, wall time, and ETA.
- **[OK]** / **[FAIL]** / **[STOP]** — when a run finishes.
- **[XFER]** — whether that run's checkpoint/logs moved from compute to login node.

Check status from the lab machine:

```bash
bash scripts/hpc_launch.sh --status
```

This shows whether the tunnel, relay, both watchers, the compute-node forwarder, the login-node
collector, and the PBS job are alive — it handles the two-hop `qstat` for you.

Watch logs on the HPC (two hops — login, then compute):

```bash
# PBS array queue:
ssh -t YOUR_USER@THE_HOST 'ssh hpc "qstat -t $(cat ~/prism/logs/hpc_jobid)"'

# A specific run's log:
ssh YOUR_USER@THE_HOST 'ssh hpc "tail -f ~/prism/logs/train_*.log"'

# The compute-node forwarder (detach with Ctrl-b then d):
ssh YOUR_USER@THE_HOST 'ssh hpc "tmux attach -t prism_forwarder"'

# The login-node collector:
ssh YOUR_USER@THE_HOST 'tmux attach -t prism_collector'
```

---

## Step 8 — Pull results back

When all runs are done (check `qstat` shows them all finished):

```bash
bash scripts/hpc_pull_results.sh
```

Then push the wandb runs to the server:

```bash
wandb sync wandb/offline-run-*
```

Verify you got the expected number of checkpoints:

```bash
find model -name '*.pt' | wc -l
```

---

## Emergency stop

```bash
bash scripts/hpc_launch.sh --stop
```

This kills the local relay and both watchers; kills the login→compute tunnel and the login-node
collector tmux session; kills the compute-node forwarder tmux session; and `qdel`s both the
smoke and full PBS jobs. All array elements terminate.

---

## Troubleshooting

**"cannot ssh to ..."** — Passwordless key auth is not set up. Go back to Step 2.

**"login node cannot reach the compute node"** — The second hop (`ssh hpc` from the login node)
is not set up passwordlessly, or `HPC_INNER_HOST` is wrong. Run `bash scripts/hpc_preflight.sh` —
probe 2 dumps the login node's `~/.ssh/config` entry for the name you configured. Ask your HPC
administrator to fix key auth between the two HPC nodes.

**Rsync stalls** — Your network probably dropped. Reconnect and re-run `bash scripts/hpc_launch.sh`
— it resumes, and it also skips any pushes (repo/.venv/data) that already landed intact on both
the login and compute nodes.

**qsub says "no matching queue"** — Wrong `HPC_QUEUE` in the config. Run `ssh -t YOUR_USER@THE_HOST 'ssh hpc "qstat -q"'` and pick a valid GPU queue.

**qsub says "resources not available"** — Compute nodes are busy. PBS will schedule when slots free.

**"qsub: command not found"** — Expected if you ran `qsub` by hand on the *login* node — it is
not there; you must be on the compute node. `scripts/hpc_common.sh`'s `compute_ssh` handles this
internally for the launcher and watchers.

**"shipped .venv failed to import torch/wandb"** — Re-run `bash scripts/hpc_launch.sh`, then
`bash scripts/hpc_preflight.sh` (probe 4) to confirm. If the venv's base Python (check
`.venv/pyvenv.cfg`'s `home =` line) does not exist on the compute node, the venv needs to be
rebuilt for that cluster's Python, or `USE_SHIPPED_VENV` needs to go to `0` for the
offline-wheels path.

**"reverse tunnel failed to start"** — Either your lab→HPC SSH key has a passphrase (the
backgrounded tunnel cannot type it), or the login node blocks port-forwarding. For the first:
make a passphraseless key, or run `eval $(ssh-agent); ssh-add ~/.ssh/id_ed25519` once, then
re-launch. For the second, contact your HPC administrator — it requires a config change on the
cluster side.

**"tunnel is up but the compute node could not reach the relay"** — Telegram messages are not
lost — they queue on the compute node in `logs/notify_queue.jsonl` and flush automatically once
the tunnel works.

**No Telegram messages arriving** — Check `logs/relay.log` and `bash scripts/hpc_launch.sh --status`. If a tunnel died, re-run the launcher — it restarts both.

**Smoke failed** — Read the Telegram failure message (the log tail is fetched directly from the compute node). Fix the issue, then re-run `bash scripts/hpc_launch.sh`.

**A single run failed but others passed** — The other runs continue independently. Decide whether to re-run the failed slot.

**Results not showing up on the lab machine despite [OK]** — Results move compute→login either immediately (if `PUSH_RESULTS_FROM_JOB=1`) or within `COLLECTOR_INTERVAL` seconds via the login-node collector. Then the lab-side grid watcher pulls login→lab on its poll cycle (`GRID_POLL_INTERVAL`, default 600s). A few-minutes lag between "[OK]" and the file appearing locally is normal.

---

## Summary

1. `git checkout hpc && git pull origin hpc`.
2. One-time only: install packages, SSH keys to **both** HPC nodes, `wandb login`.
3. Fill `scripts/hpc_config.env` (every `FILL_ME`, including `HPC_INNER_HOST`).
4. Confirm data exists at `LAB_DATA_ROOT`.
5. `bash scripts/hpc_preflight.sh` — fix anything marked `FAIL`.
6. `bash scripts/hpc_launch.sh` — sit for 1–1.5 h if the data push actually runs (skipped if already present on both remote nodes).
7. Wait for Telegram messages.
8. `bash scripts/hpc_pull_results.sh` + `wandb sync wandb/offline-run-*`.
