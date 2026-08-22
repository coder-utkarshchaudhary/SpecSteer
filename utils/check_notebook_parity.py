"""
utils/check_notebook_parity.py
------------------------------
Prove the four Kaggle notebooks are safe to commit, without a GPU.

WHY
===
`notebooks/*.ipynb` inline a copy of `utils/config.py`, `utils/hyperparams.py`
and all four hyperparam YAMLs so they run standalone on Kaggle. CLAUDE.md
already records what that costs: `CRIMS: 544` survived in five places at once
because nothing checked. "Double-check before committing" is not a deliverable;
this is.

CHECKS
======
1. Every code cell parses (``ast.parse``). Catches the syntax and indentation
   errors that a notebook only reveals when a human runs the cell.
2. Cells that are supposed to be byte-identical across all four notebooks still
   are — the config cell and the training-loop cell. A change applied to three
   of four is the exact failure mode this guards.
3. Every inlined hyperparameter equals the repo YAML for that dataset, except
   for a whitelist of deliberate divergences (below).
4. The inlined band counts equal `utils/config.py`'s.
5. With ``--execute``: every notebook is RUN end to end on synthetic data (CPU,
   one dataset, one seed, one epoch), with only the data layer stubbed. Syntax
   checking cannot catch a wrong variable name inside a loop body or a tuple
   that gained a field in one place and not another; running it can.

DELIBERATE DIVERGENCES (not failures)
=====================================
* ``num_workers``  — 4 in the notebooks, 8 in the YAMLs. Kaggle has 2-4 usable
  cores.
* ``batch_size``   — the notebooks are sized for Kaggle's 2x15 GB, the YAMLs for
  the lab's 20 GB. This IS a within-dataset confound when one dataset is trained
  on both platforms; it is recorded per checkpoint and flagged in VERDICT.txt
  rather than silently tolerated. See Part F of the plan.

Usage:
    PYTHONPATH=. python utils/check_notebook_parity.py
    PYTHONPATH=. python utils/check_notebook_parity.py --show-batch
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from utils.config import DATASETS  # noqa: E402

NOTEBOOKS = ["vae-our", "vae-standard", "vae-3d-spatio-spectral", "vae-1d-pixelwise"]

# Cell indices of the cells that must match across notebooks. vae-our has three
# extra model cells, so its indices are shifted after cell 12.
SHARED_CELLS = {
    "config":   {"vae-our": 2,  "vae-standard": 2,  "vae-3d-spatio-spectral": 2,  "vae-1d-pixelwise": 2},
    "training": {"vae-our": 20, "vae-standard": 17, "vae-3d-spatio-spectral": 17, "vae-1d-pixelwise": 17},
}

# Keys allowed to differ from the YAML, with the reason.
WHITELIST = {
    "num_workers": "Kaggle has 2-4 usable cores; YAML targets the lab",
    "batch_size": "notebook tier is sized for Kaggle 2x15 GB, YAML for lab 20 GB",
}

# Hyperparam keys that MUST agree.
CHECKED = [
    "epochs", "lr", "beta", "lambda_physics", "seed", "weight_decay",
    "early_stopping_patience",
    "spectral_latent_dim", "vae_standard_latent_ch", "vae_3d_latent_ch",
    "vae_1d_latent_dim",
    "vae_standard_base_ch", "vae_3d_base_ch", "vae_1d_hidden_dims",
]


def load_nb(name: str) -> dict:
    return json.loads((REPO_ROOT / "notebooks" / f"{name}.ipynb").read_text())


def cell_src(nb: dict, i: int) -> str:
    return "".join(nb["cells"][i]["source"])


def exec_config_cell(src: str) -> dict:
    """
    Run just the inlined config cell and hand back its namespace.

    The cell is pure data plus dataclass/function definitions — no torch, no I/O
    — so executing it is how we read the values the notebook will ACTUALLY use,
    rather than regex-scraping literals and hoping the parse matches Python's.
    """
    ns: dict = {}
    exec(compile(src, "<config-cell>", "exec"), ns)   # noqa: S102 - trusted repo file
    return ns


def check_syntax(problems: list[str]) -> None:
    for name in NOTEBOOKS:
        nb = load_nb(name)
        for i, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell["source"])
            try:
                ast.parse(src)
            except SyntaxError as e:
                problems.append(
                    f"{name}.ipynb cell {i}: SyntaxError line {e.lineno}: {e.msg}\n"
                    f"      {(e.text or '').rstrip()}"
                )


def check_shared_cells(problems: list[str]) -> None:
    for label, idx in SHARED_CELLS.items():
        digests = {}
        for name in NOTEBOOKS:
            src = cell_src(load_nb(name), idx[name])
            digests.setdefault(hashlib.md5(src.encode()).hexdigest(), []).append(name)
        if len(digests) > 1:
            groups = " | ".join(f"{d[:8]}: {','.join(v)}" for d, v in digests.items())
            problems.append(
                f"the '{label}' cell is NOT identical across notebooks -> {groups}\n"
                f"      an edit was applied to some but not all four."
            )


def check_values(problems: list[str], show_batch: bool) -> None:
    ns = exec_config_cell(cell_src(load_nb(NOTEBOOKS[0]), SHARED_CELLS["config"]["vae-our"]))
    nb_hp = ns["HYPERPARAMS"]
    nb_ds = ns["DATASETS"]

    for ds in sorted(DATASETS):
        ypath = REPO_ROOT / "utils" / "hyperparam_configs" / f"hyperparam-config-{ds}.yaml"
        y = yaml.safe_load(ypath.read_text())
        n = nb_hp[ds]

        for key in CHECKED:
            if key not in y:
                continue
            yv, nv = y[key], n.get(key)
            if isinstance(yv, list):
                yv = tuple(yv)
            if isinstance(nv, list):
                nv = tuple(nv)
            if yv != nv:
                problems.append(
                    f"{ds}.{key}: notebook={nv!r} but {ypath.name}={yv!r}"
                )

        # band counts
        if nb_ds[ds]["input_channels"] != DATASETS[ds]["input_channels"]:
            problems.append(
                f"{ds}.input_channels: notebook={nb_ds[ds]['input_channels']} "
                f"but utils/config.py={DATASETS[ds]['input_channels']}"
            )

        if show_batch:
            print(f"  {ds:<8} notebook batch {n['batch_size']:>3} "
                  f"({n['batch_size'] // 2:>2}/device on 2 GPUs)   "
                  f"script batch {y['batch_size']:>3}")


def execute_notebook(name: str) -> tuple[bool, str]:
    """
    Run one notebook's cells with a synthetic data layer.

    Only `build_dataloader` and `CKPT_ROOT` are replaced; everything else is the
    notebook's own code. The driver is narrowed to one dataset / one seed / one
    epoch, because the question is "does this run", not "does this converge".
    """
    import contextlib
    import io
    import tempfile

    import torch

    cells = load_nb(name)["cells"]
    code = [i for i, c in enumerate(cells) if c.get("cell_type") == "code"]
    driver = code[-1]
    ns: dict = {"__name__": "__main__"}
    buf = io.StringIO()
    ckpt_root = tempfile.mkdtemp()

    try:
        with contextlib.redirect_stdout(buf):
            for i in code[:-1]:
                exec(compile("".join(cells[i]["source"]), f"cell{i}", "exec"), ns)  # noqa: S102

            chan = {d: ns["DATASETS"][d]["input_channels"] for d in ns["DATASETS"]}

            def fake_loader(root, split, shuffle=False, batch_size=None):
                c = chan[Path(root).name]
                n = 2 if split == "train" else 1
                return [torch.rand(2, 64, 64, c) for _ in range(n)]

            ns["build_dataloader"] = fake_loader
            ns["CKPT_ROOT"] = ckpt_root

            src = "".join(cells[driver]["source"])
            import re as _re
            src = _re.sub(r"RUN_DATASETS = \[[^\]]*\]", 'RUN_DATASETS = ["M3"]', src)
            src = src.replace("RUN_SEEDS = SEEDS", "RUN_SEEDS = [42]")
            src = src.replace("settings.epochs", "1")
            exec(compile(src, "driver", "exec"), ns)  # noqa: S102
    except Exception as e:                      # noqa: BLE001 - report, don't raise
        tail = "\n".join(buf.getvalue().splitlines()[-3:])
        return False, f"{type(e).__name__}: {e}" + (f"\n      last output: {tail}" if tail else "")

    saved = sorted(Path(ckpt_root).rglob("*.pt"))
    if not saved:
        return False, "ran, but wrote no checkpoint"
    per_cell = len(ns.get("LOSS_TYPES", ["physics"]))
    if len(saved) != 2 * per_cell:
        return False, (f"expected {2 * per_cell} checkpoints "
                       f"(2 per cell x {per_cell} loss regimes), got {len(saved)}")
    return True, f"{len(saved)} checkpoints, e.g. {saved[0].name}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify notebooks match the repo config.")
    ap.add_argument("--show-batch", action="store_true",
                    help="Print the two batch tiers side by side.")
    ap.add_argument("--execute", action="store_true",
                    help="Also RUN each notebook on synthetic data (CPU, ~1 min "
                         "total). Catches logic errors that parsing cannot.")
    args = ap.parse_args()

    problems: list[str] = []
    print("checking notebook syntax ...")
    check_syntax(problems)
    print("checking shared cells are identical ...")
    check_shared_cells(problems)
    print("checking inlined values against the repo ...")
    if args.show_batch:
        print("\n  batch tiers (deliberately different):")
    check_values(problems, args.show_batch)

    if args.execute:
        print("executing each notebook on synthetic data ...")
        for name in NOTEBOOKS:
            ok, detail = execute_notebook(name)
            print(f"  {'ok  ' if ok else 'FAIL'}  {name:<24} {detail}")
            if not ok:
                problems.append(f"{name}.ipynb failed to execute: {detail}")

    print()
    if problems:
        print(f"FAILED — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        print("\nFix the notebooks (or the YAMLs) before committing.")
        return 1

    print("OK — all four notebooks parse, the shared cells are identical, and every")
    print("     inlined value matches the repo except the whitelisted divergences:")
    for k, why in WHITELIST.items():
        print(f"       {k}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
