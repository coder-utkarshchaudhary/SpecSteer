"""
utils/dataset/inspect_channels.py
---------------------------------
Print, for every dataset, the band count that is actually on disk and check it
against ``utils/config.py``. Exits non-zero on any mismatch.

This exists because a wrong constant here is catastrophically expensive: CRIMS
was configured as 544 bands while its patches are 457, so every CRIMS run in the
28-slot grid died in ``build_dataloader`` before ``RunNotifier`` was even
constructed — seven silent failures with no Telegram message and no log line.

The check is **crop-aware**. Three numbers have to line up:

    raw C on disk  ==  DATASETS[ds]["raw_channels"]
    crop_bands (or raw C if unset)  ==  DATASETS[ds]["input_channels"]
    input_channels % 2**spectral_n_1D_conv_blocks  ==  0

The last one is what makes CRIMS need a crop at all: 457 is prime, so the
spectral encoder's two stride-2 halvings (457 -> 114) can never be undone
exactly (114 -> 456 != 457). Cropping to 456 fixes it, exactly as M3 crops
85 -> 84.

Usage:
    PYTHONPATH=. python utils/dataset/inspect_channels.py
    PYTHONPATH=. python utils/dataset/inspect_channels.py --dataset CRIMS
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import (  # noqa: E402
    DATASETS,
    effective_channels,
    probe_channels,
    settings,
)


def suggest_crop(raw_c: int, n_blocks: int) -> int:
    """Largest value <= raw_c that survives the spectral round-trip exactly."""
    mult = 2 ** n_blocks
    return raw_c - (raw_c % mult)


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify configured vs on-disk band counts.")
    ap.add_argument("--dataset", action="append", choices=sorted(DATASETS),
                    help="Dataset to check (repeatable). Default: all.")
    ap.add_argument("--processed-root", default=None,
                    help="Override the search root for the single --dataset given.")
    args = ap.parse_args()

    datasets = args.dataset or sorted(DATASETS)
    if args.processed_root and len(datasets) > 1:
        raise SystemExit("--processed-root requires exactly one --dataset")

    n_blocks = settings.spectral_n_1D_conv_blocks
    mult = 2 ** n_blocks

    header = (f"{'dataset':<8} {'source':<8} {'raw C':>6} {'crop':>6} {'eff C':>6} "
              f"{'cfg C':>6} {'%' + str(mult):>5}  status")
    print(header)
    print("-" * len(header))

    problems: list[str] = []
    missing: list[str] = []

    for ds in datasets:
        meta = DATASETS[ds]
        packed = Path(meta["packed_root"]) / "train.npy"
        source = "packed" if packed.is_file() else "raw"

        raw_c = probe_channels(ds, args.processed_root)
        if raw_c is None:
            print(f"{ds:<8} {'-':<8} {'-':>6} {'-':>6} {'-':>6} "
                  f"{meta['input_channels']:>6} {'-':>5}  NO DATA (not staged here)")
            missing.append(ds)
            continue

        # A packed shard is already cropped; a raw patch is not.
        eff = int(raw_c) if source == "packed" else effective_channels(ds, raw_c)
        crop = meta.get("crop_bands")
        cfg_c = meta["input_channels"]
        divisible = (eff % mult) == 0

        notes = []
        if source == "raw":
            expected_raw = meta.get("raw_channels")
            if expected_raw is not None and raw_c != expected_raw:
                notes.append(f"raw C {raw_c} != raw_channels {expected_raw}")
        if eff != cfg_c:
            notes.append(f"effective C {eff} != input_channels {cfg_c}")
        if not divisible:
            notes.append(
                f"{eff} not divisible by {mult}; set crop_bands="
                f"{suggest_crop(eff, n_blocks)}"
            )

        status = "OK" if not notes else "MISMATCH -> " + "; ".join(notes)
        if notes:
            problems.append(f"{ds}: {'; '.join(notes)}")

        print(f"{ds:<8} {source:<8} {raw_c:>6} {str(crop or '-'):>6} {eff:>6} "
              f"{cfg_c:>6} {'yes' if divisible else 'NO':>5}  {status}")

    print()
    if missing:
        print(f"not staged on this machine: {', '.join(missing)} "
              f"(nothing to verify — not an error)")
    if problems:
        print("FAILED:")
        for p in problems:
            print(f"  - {p}")
        print("\nFix utils/config.py DATASETS, then re-run. If the crop changed, "
              "re-pack with: python utils/dataset/pack.py --overwrite")
        return 1

    print("all staged datasets agree with utils/config.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
