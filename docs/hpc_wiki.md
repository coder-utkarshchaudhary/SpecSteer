# Running the Ablation Grid on IITD HPC

This document walks you through everything from "I just sat down at the lab machine" to "checkpoints are back and Utkarsh is notified." Read it top-to-bottom like a recipe — every step depends on the ones before it.

You will touch exactly two scripts: **`bash scripts/hpc_preflight.sh`** (read-only checks, run first) and **`bash scripts/hpc_launch.sh`** (does the actual work). Everything else is automated. If something breaks and this doc doesn't cover it, don't improvise — call Utkarsh.

---

## The two-node topology (read this once, it explains everything below)

IITD's HPC is actually **two separate machines with two separate filesystems**:

- The **login node** — what `HPC_HOST` in the config points at. Reached directly from the
  lab as `ssh ${HPC_USER}@${HPC_HOST}`. This machine does **not** have `qsub` — it's purely a
  staging/jump host.
- The **compute node** — reached by first `ssh`-ing into the login node, and from *there*
  running `ssh hpc` (or whatever `HPC_INNER_HOST` is set to in the config). This is where jobs
  actually run and where `qsub`/`qstat`/`qdel` live.

Everything in `scripts/hpc_launch.sh` therefore happens in two hops: repo/venv/data get pushed
lab → login → compute; results come back compute → login → lab; Telegram notifications go
lab → login → compute (a tunnel) and compute → login → lab (the messages themselves). You never
have to manage this by hand — `scripts/hpc_common.sh` wraps both hops for every script — but
when something looks stuck, knowing which hop it's stuck on is the first diagnostic question.

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

**Also verify the second hop** — once you're logged into the login node, `ssh hpc` (or whatever
`HPC_INNER_HOST` is set to) must work *without a password* too, since that's how the launcher
reaches the compute node where `qsub` actually lives:

```bash
ssh YOUR_HPC_USER@THE_HPC_HOST 'ssh hpc echo ok'
```

If that doesn't print `ok`, ask Kavinder — the login node's own SSH key needs to be authorized
on the compute node, which is a one-time setup on the HPC side, not something this repo's
scripts can fix.

Finally, log in to Weights & Biases on the lab machine (Utkarsh will give you the API key privately) using a new terminal window. This repo's Python environment lives in `.venv/` (a plain venv, not conda) — activate it first:

```bash
source .venv/bin/activate
pip install wandb   # only if wandb isn't already in .venv
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
- `HPC_INNER_HOST` — what you type after `ssh` once you're already logged into the login node to
  reach the compute node (default `hpc`). Verify with
  `ssh YOUR_USER@THE_HOST 'ssh HPC_INNER_HOST echo ok'` — see "Before you begin" above.

**PBS queue (ask Utkarsh):**

- `HPC_QUEUE` — which GPU queue to submit to. If you want to check yourself: `ssh -t YOUR_USER@THE_HOST 'qstat -q'` lists them (the `-t` forces a login shell so `qstat` is on `PATH`). Pick the one with A100 access. When in doubt, ask Utkarsh.

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

You should see subdirectories: `IIRS`, `M3`, `AVIRIS`, `CRIMS` (exact case — a lowercase `crims` or `m3` will silently fail the on-HPC data check later). If any are missing, **stop and call Utkarsh** — do not proceed without the full dataset.

---

## Step 5 — Preflight (read-only, always run this first)

```bash
bash scripts/hpc_preflight.sh
```

This checks, without changing anything: that you can reach the login node, that the login node
can reach the compute node, that both nodes have the tools the launcher needs, and — if this
repo's `.venv/` has already been pushed to the HPC — that it actually imports `torch` and
`wandb` there. Fix anything it reports as `FAIL` before continuing; `WARN` lines are fine to
proceed past (they're usually "not pushed yet," which is expected on a first run).

---

## Step 6 — Launch
In a new terminal window run the following command from repo root:
```bash
bash scripts/hpc_launch.sh
```

That's it. The script does everything automatically:

1. Checks WiFi, config, SSH access to both nodes, local data.
2. Skips the pip-wheel build entirely (this repo ships a prebuilt `.venv/` — see
   `USE_SHIPPED_VENV` in the config; only set that to `0` if Utkarsh tells you to).
3. Rsyncs the repo, `.venv/`, and processed data from the **lab to the login node** — each of
   the three is pushed independently and only if it's missing or looks broken on the login node
   (**this is the slow step when it does run — 1 to 3 hours over lab WiFi for the data**).
4. Rsyncs the same three things again, this time from the **login node to the compute node**
   (the two machines don't share a filesystem).
5. Runs the HPC bootstrap **on the compute node** (verifies the shipped `.venv` works; only
   falls back to an offline pip install if you set `USE_SHIPPED_VENV=0`).
6. Starts the Telegram relay on the lab machine, a reverse tunnel chained lab → login → compute,
   a message forwarder in tmux **on the compute node**, and a results collector in tmux **on the
   login node**. (Opening the tunnel might ask for the lab system password — if it does, ask any
   faculty.)
7. Submits a **smoke run** — one slot, 5 epochs — to verify the environment works.
8. Starts a background watcher that waits for the smoke to finish.

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

## Step 7 — Monitor

Utkarsh will receive Telegram messages throughout the run:

- **[START]** — one per run, showing the hyperparameters.
- **[HB]** — every 10 epochs, with current loss / MSE / SAM / KLD metrics, wall time, and ETA.
- **[OK]** / **[FAIL]** / **[STOP]** — when a run finishes.
- **[XFER]** — whether that run's checkpoint/logs made it from the compute node back to the
  login node (only appears if `PUSH_RESULTS_FROM_JOB=1`; otherwise the login-node collector
  moves results on its own timer and you won't see this per-run).

If you want to check status from the lab machine:

```bash
bash scripts/hpc_launch.sh --status
```

This shows whether the tunnel, relay, both watchers, the compute-node forwarder, the login-node
collector, and the PBS job are alive — it handles the two-hop `qstat` for you.

If you want to watch logs on the HPC, you now need two hops (login, then compute):

```bash
# PBS array queue:
ssh -t YOUR_USER@THE_HOST 'ssh hpc "qstat -t \$(cat ~/prism/logs/hpc_jobid)"'

# A specific run's log (lives on the COMPUTE node, pushed/pulled to login over time):
ssh YOUR_USER@THE_HOST 'ssh hpc "tail -f ~/prism/logs/train_*.log"'

# The compute-node forwarder (detach with Ctrl-b then d):
ssh YOUR_USER@THE_HOST 'ssh hpc "tmux attach -t prism_forwarder"'

# The login-node collector:
ssh YOUR_USER@THE_HOST 'tmux attach -t prism_collector'
```

---

## Step 8 — Pull results back

When all 28 runs are done (you'll get 28 `[OK]` messages on Telegram, or check `qstat` shows them all finished — the login-node collector will already have pulled most of them back automatically as each slot finished):

```bash
bash scripts/hpc_pull_results.sh
```

This rsyncs the checkpoints, wandb offline runs, and logs from the HPC **login node** to the lab
machine (final catch-all sweep — the grid watcher already did this per-slot as runs finished).

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

This kills the local relay and both watchers; kills the login→compute tunnel and the login-node
collector tmux session; kills the compute-node forwarder tmux session; and `qdel`s both the
smoke and full PBS jobs (via the two-hop `compute_ssh`). All array elements terminate.

---

## Troubleshooting

**"cannot ssh to ..."** — Passwordless key auth isn't set up. Go back to Step 2.

**"login node cannot reach the compute node"** — The second hop (`ssh hpc` from the login node) isn't set up passwordlessly, or `HPC_INNER_HOST` is wrong. Run `bash scripts/hpc_preflight.sh` — probe 2 dumps the login node's `~/.ssh/config` entry for the name you configured. Ask Kavinder to fix key auth between the two HPC nodes; this repo's scripts can't do that part for you.

**Rsync stalls** — Reconnect to `mlr lab 5g`. Re-run `bash scripts/hpc_launch.sh` — it resumes, and it also skips any of the three pushes (repo/.venv/data) that already landed intact, on both the login *and* the compute node.

**qsub says "no matching queue"** — Wrong `HPC_QUEUE` in the config. Run `ssh -t YOUR_USER@THE_HOST 'ssh hpc "qstat -q"'` and pick a valid GPU queue. Ask Utkarsh.

**qsub says "resources not available"** — A100 nodes are busy. PBS will schedule when slots free. Nothing to do.

**"qsub: command not found"** — Expected if you ran `qsub` by hand on the *login* node — it genuinely isn't there; you have to be on the compute node. From the login node: `ssh hpc 'qstat -q'`. If you're already on the compute node and still see this, PBS Pro's scheduler binaries are only put on `PATH` by the login profile (`/etc/profile.d`), which a plain non-interactive `ssh HOST 'cmd'` doesn't source — `scripts/hpc_common.sh`'s `compute_ssh` handles this internally (login-shell + a `PBS_EXEC` PATH fallback), so the launcher/watchers never hit it; only matters if you're running commands by hand.

**"shipped .venv failed to import torch/wandb"** — Almost always means the `.venv` push got its excludes wrong and silently dropped a site-packages subfolder (the venv's `wandb/`, `logs/`, `model/` etc. subdirectories share names with repo-root excludes). `scripts/hpc_launch.sh` now anchors those excludes and gives `.venv/` its own rsync pass, so a fresh push should fix it — re-run `bash scripts/hpc_launch.sh`, then `bash scripts/hpc_preflight.sh` (probe 4) to confirm. If the venv's base Python (check `.venv/pyvenv.cfg`'s `home =` line) genuinely doesn't exist on the compute node, tell Utkarsh — the venv needs to be rebuilt for that cluster's Python, or `USE_SHIPPED_VENV` needs to go back to `0` for the offline-wheels path.

**"reverse tunnel failed to start"** — Two causes for the lab→login leg. Either your lab→HPC SSH key has a passphrase (the backgrounded tunnel can't type it), or the login node blocks port-forwarding. For the first: make a passphraseless key, or run `eval $(ssh-agent); ssh-add ~/.ssh/id_ed25519` once, then re-launch. For the second, call Utkarsh — it needs a config change on his side. Note: the tunnel logs in *to the HPC* using your HPC key — it never asks for the lab machine's password. There's now a second, inner leg (login→compute) supervised in a login-node tmux session (`prism_inner_tunnel`) — if Telegram works but is delayed, that's usually the one that needs a poke; `bash scripts/hpc_launch.sh --status` shows both.

**"tunnel is up but the compute node could not reach the relay"** — The launcher warns about this but keeps going. Telegram messages are not lost — they queue on the compute node in `logs/notify_queue.jsonl` and flush automatically once the tunnel works. If messages never arrive, one of the two hops is blocking forwarding; tell Utkarsh.

**No Telegram messages arriving** — Check `logs/relay.log` and `bash scripts/hpc_launch.sh --status`. If a tunnel died, just re-run the launcher — it restarts both. Note: if `bash scripts/hpc_preflight.sh` (probe 5) found the compute node has direct outbound internet, the launcher skips the whole tunnel/relay/forwarder/collector setup and Telegram goes straight from the compute node — in that mode `--status` won't show a tunnel and that's correct, not a failure.

**Smoke failed** — Read the Telegram failure message (its log tail is fetched directly from the compute node, no rsync needed). Screenshot it and send to Utkarsh. Do not attempt to fix code yourself.

**"wheels/ empty" during bootstrap** — Only relevant if `USE_SHIPPED_VENV=0` in the config (the shipped-venv path, the default, never touches wheels/). If you did set it to `0`: the `PIP_PYTHON_VERSION` in the config doesn't match the HPC's python. Run `ssh YOUR_USER@THE_HOST 'ssh hpc "python3 --version"'` and update `PIP_PYTHON_VERSION` and `PIP_ABI` to match.

**A single run failed but others passed** — That's expected for edge cases. The other 27 runs continue independently. Tell Utkarsh about the failure; he'll decide whether to re-run it.

**Results aren't showing up on the lab machine even though Telegram says [OK]** — Results move compute→login either immediately (if `PUSH_RESULTS_FROM_JOB=1` and that push succeeds) or within `COLLECTOR_INTERVAL` seconds (default 120s) via the login-node collector tmux session, which is the fallback path and the one that's always eventually correct. Then the lab-side grid watcher pulls login→lab on its own poll cycle (`GRID_POLL_INTERVAL`, default 600s). So there can be a few-minutes lag between "[OK]" and the file appearing locally — that's normal, not a bug. If it's been much longer, check `ssh YOUR_USER@THE_HOST 'tmux attach -t prism_collector'` for errors.

---

## Summary

1. Connect to `mlr lab 5g`.
2. `git checkout hpc && git pull origin hpc`.
3. One-time only: install packages, SSH keys to **both** HPC nodes, `wandb login`.
4. Fill `scripts/hpc_config.env` (ask Utkarsh / Kavinder for every `FILL_ME`, including `HPC_INNER_HOST`).
5. Confirm data exists at `LAB_DATA_ROOT`.
6. `bash scripts/hpc_preflight.sh` — fix anything marked `FAIL`.
7. `bash scripts/hpc_launch.sh` — sit for 1–1.5 h if the data push actually runs (it's skipped if already present on both remote nodes).
8. Wait for Telegram messages.
9. `bash scripts/hpc_pull_results.sh` + `wandb sync wandb/offline-run-*`.
10. Tell Utkarsh.

If it's not in this list, don't do it. If something fails, call Utkarsh.
