from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """
    Centralised configuration for the Dual-Stream Physics-Informed VAE pipeline.

    All derived / computed fields are populated in __post_init__, so only the
    primary fields need to be set when instantiating Settings().

    Spectral arithmetic (with spectral_conv1D_kernel_size=4, stride=2, padding=1):
      Conv1d:          L_out = L_in // 2
      ConvTranspose1d: L_out = 2 * L_in
    With n=2 blocks, spectral_base_ch=32 and input_channels=256 (IIRS):
      Encoder:  256 → 128 → 64  (L);  channels  1 → 32 → 64
      Decoder:   64 → 128 → 256 (L);  channels 64 → 32 → 1
    Sequence length L still tracks the band count (so the latent stays
    sensor-aware); only the channel width is now a free hyper-parameter.

    NOTE: input_channels varies per dataset (IIRS=256, M3=84, AVIRIS=424,
    CRIMS=456) — see DATASETS / make_settings() below. Every dataset needs its
    own Settings instance (and therefore its own model instance) since band
    count differs.

    PERF NOTE: the spectral branch's conv width is `spectral_base_ch`, NOT
    `input_channels`. Tying the width to the band count (as the original code
    did) made the branch cost scale as O(C^2) per pixel spectrum — 99.7% of
    vae-our's total FLOPs — and made vae-our's parameter count swing 3x across
    sensors for no principled reason. See CLAUDE.md §10.
    """

    # ------------------------------------------------------------------
    # Input patch shape (output of slice.py)
    # ------------------------------------------------------------------
    input_height: int = 64
    input_width: int = 64
    input_channels: int = 256          # whole-cube band count; overridden per dataset

    # ------------------------------------------------------------------
    # Preprocessing knobs (used by utils/dataset/preprocess.py)
    # ------------------------------------------------------------------
    # Target wavelength (nm) for the normalisation reference band. The actual
    # band index is resolved per-cube from that cube's own wavelength array
    # (nearest match to this target) — no band selection/cropping is applied
    # to the cube itself; the whole spectrum is kept.
    norm_target_nm: float = 1500.0
    savgol_window: int = 7
    savgol_polyorder: int = 2

    # Drop any patch whose fraction of fill/invalid pixels exceeds this.
    fill_fraction_threshold: float = 0.05

    # ------------------------------------------------------------------
    # Slicing knobs (used by utils/dataset/slice.py)
    # ------------------------------------------------------------------
    patch_size: int = 64
    patch_stride: int = 48            # 25% overlap: stride = patch_size * 0.75
    split_ratios: tuple = (0.70, 0.15, 0.15)   # (train, valid, test)

    # ------------------------------------------------------------------
    # Data paths (generic defaults; per-dataset paths live in DATASETS)
    # ------------------------------------------------------------------
    data_original_root: str = "data/original"
    data_processed_root: str = "data/processed"
    # Packed fp16 memmap shards written by utils/dataset/pack.py. One file per
    # (dataset, split) replaces ~15k individual .npy patches — see CLAUDE.md §4.
    data_packed_root: str = "data/packed"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    # Fallback only — the real value is per-dataset in the hyperparam YAMLs
    # (IIRS 32, M3 32, AVIRIS 16, CRIMS 16). Batch size is held constant across
    # all 4 models x 2 loss regimes WITHIN a dataset, which is the axis the
    # ablation compares on; across sensors there is no controlled comparison to
    # protect, and a single global value would idle the GPU on the lighter ones.
    # Re-derive with: python utils/find_max_batch.py --budget-gb 24 [--fit]
    batch_size: int = 16
    num_workers: int = 8

    # Cap on the number of *training* patches per dataset. Applied by
    # utils/dataset/pack.py with a fixed seed, stratified across scenes, so every
    # model sees the identical subset. valid/test splits are never capped.
    # None disables the cap.
    train_patch_cap: int | None = 7000
    # Seed for the cap's subsampling. Deliberately separate from the training
    # seed so changing one never silently reshuffles the other.
    patch_cap_seed: int = 1234

    # ------------------------------------------------------------------
    # Spatial branch (vae-our)
    # ------------------------------------------------------------------
    # Conv1D channel reduction per pixel spectrum. Block channels run
    # r -> 2r -> 4r -> 8r, so this is also the width dial of the whole spatial
    # stream — PRISM's natural capacity knob (per-dataset override in the YAMLs).
    reduced_dims: int = 64
    # Channels of the 8x8 spatial GRID latent (after the reparameterize chunk).
    # Replaces the old `latent_dim` global 256-vector: z_s is now
    # (B, vae_our_spatial_latent_ch, 8, 8) — spatially addressed, so texture no
    # longer squeezes through a single whole-patch vector (docs/new_plan.md,
    # Iteration 1). Latent budget: 64*d_s + H*W*d_p lands on the common T —
    # see utils/match_latent_rate.py.
    vae_our_spatial_latent_ch: int = 64
    n_2D_conv_blocks: int = 3        # 64 / 2^3 = 8 px grid — this IS the latent grid
    conv2D_kernel_size: int = 3

    # --- vae-our fusion (spatially-adaptive gated late fusion) ---
    # Hidden width of the 2x(3x3 conv) gate network in modules/vae_our.py.
    vae_our_fusion_hidden: int = 64
    # Ablation flag: False falls back to the old global Linear(2C -> C) fusion.
    vae_our_adaptive_fusion: bool = True
    # Weight of each auxiliary per-stream MSE relative to the fused-recon MSE.
    # The mix is 0.5 : w : w, normalised to sum to 1 in loss_terms() so the
    # reconstruction term stays on the baselines' scale. It.3 sweeps {0.05, 0.1}.
    vae_our_aux_mse_weight: float = 0.1

    # Derived spatial (computed in __post_init__)
    conv_output_c: int = field(init=False)
    conv_output_h: int = field(init=False)
    conv_output_w: int = field(init=False)

    # ------------------------------------------------------------------
    # Spectral branch
    # ------------------------------------------------------------------
    spectral_n_1D_conv_blocks: int = 2
    # k=4 paired with stride=2, pad=1 gives exact halving/doubling:
    #   Conv1d:          L_out = L_in // 2
    #   ConvTranspose1d: L_out = 2 * L_in
    spectral_conv1D_kernel_size: int = 4
    spectral_latent_dim: int = 128    # per-pixel spectral latent (after reparameterize chunk)
    # Conv1d width of the spectral branch's first block; doubles each block.
    # Mirrors `reduced_dims` on the spatial branch. Decoupled from
    # input_channels — see the class docstring's PERF NOTE.
    spectral_base_ch: int = 32

    # Derived spectral (computed in __post_init__)
    spectral_linear_expansion_dim: int = field(init=False)
    spectral_transpose_c: int = field(init=False)
    spectral_transpose_l: int = field(init=False)

    # ------------------------------------------------------------------
    # Baseline capacity knobs — REFERENCE WIDTHS (protocol change 2026-09-04).
    #
    # Baselines are no longer parameter-matched to vae-our. Each runs at the
    # width of its citable reference implementation, identical across datasets:
    #   vae-standard : base_ch 128, n_down 3 (128/256/512) — AutoencoderKL,
    #                  Rombach et al., CVPR 2022 (latent-diffusion f8 encoder).
    #   vae-3d       : base_ch 24 — compact 3-D conv stacks per the 3D-CAE of
    #                  Mei et al., IEEE TGRS 2019 (representative default;
    #                  swap in the exact per-layer counts if the PDF surfaces).
    #   vae-1d       : hidden (512, 256, 128) — 4 fully-connected layers as in
    #                  per-pixel spectral (V)AEs, e.g. Palsson et al., IEEE
    #                  Access 2018; Liu et al., IEEE TGRS 2022.
    # The capacity confound is handled post-hoc instead: a params column in
    # every table plus capacity points (each competitive baseline re-trained at
    # ~vae-our's param count, seed 42) — solve those with:
    #     python utils/check-model-params.py --solve-capacity
    # The LATENT-RATE knobs (below / per-dataset YAML) remain the hard control.
    # ------------------------------------------------------------------
    vae_standard_base_ch: int = 128
    vae_standard_n_down: int = 3
    vae_standard_latent_ch: int = 16

    vae_3d_base_ch: int = 24
    vae_3d_n_down: int = 3
    vae_3d_latent_ch: int = 8

    vae_1d_hidden_dims: tuple = (512, 256, 128)
    vae_1d_latent_dim: int = 32

    def __post_init__(self):
        # ---- Spatial derived fields ----
        self.conv_output_c = self.reduced_dims * (2 ** self.n_2D_conv_blocks)
        self.conv_output_h = self.input_height // (2 ** self.n_2D_conv_blocks)
        self.conv_output_w = self.input_width // (2 ** self.n_2D_conv_blocks)

        # ---- Spectral derived fields ----
        # Encoder Conv1d final channel count:
        #   Block i goes from (in_c → out_c) where out_c doubles each step
        #   starting at spectral_base_ch.
        #   After n blocks: final channel = spectral_base_ch * 2^(n-1)
        self.spectral_transpose_c = self.spectral_base_ch * (
            2 ** (self.spectral_n_1D_conv_blocks - 1)
        )

        # Encoder Conv1d final sequence length (halved each block):
        #   L_final = input_channels // 2^n
        self.spectral_transpose_l = self.input_channels // (
            2 ** self.spectral_n_1D_conv_blocks
        )

        # Linear expansion size (what the decoder's linear layer must produce):
        self.spectral_linear_expansion_dim = (
            self.spectral_transpose_c * self.spectral_transpose_l
        )


settings = Settings()   # default: IIRS-shaped (input_channels=256)


# ---------------------------------------------------------------------------
# Multi-dataset registry
# ---------------------------------------------------------------------------
# Each entry describes one sensor's raw location, processed output location,
# native band count (after any dataset-specific cropping needed to keep the
# spectral branch's Conv1d/ConvTranspose1d arithmetic exact), and fill value.
#
# Band counts:
#   IIRS   : 256 bands, divides cleanly by 2^spectral_n_1D_conv_blocks (4) → no crop.
#   M3     : 85 bands natively — NOT divisible by 4 (85 → 21 → 84 ≠ 85 on the
#            decoder round-trip). Cropped to 84 (drop the last band).
#   AVIRIS : 424 bands, divides cleanly by 4 → no crop.
#   CRIMS  : 457 bands on disk — 457 is prime, so the spectral round-trip can
#            never be exact (457 // 4 = 114 → 456 ≠ 457). Cropped to 456,
#            exactly as M3 crops 85 → 84. Already-preprocessed patches
#            delivered to data/processed/CRIMS/ via Google Drive.
#
# `crop_bands` is applied by utils/dataset/slice.py for the sensors it
# preprocesses, and by utils/dataset/pack.py for every sensor (including CRIMS,
# which slice.py never sees). `raw_channels` records what is actually on disk so
# the crop-aware verification in inspect_channels() has something to compare to.
DATASETS = {
    "IIRS": {
        "input_channels": 256,
        "raw_channels": 256,
        "raw_root": "data/original - IIRS",
        "processed_root": "data/processed/IIRS",
        "packed_root": "data/packed/IIRS",
        "fill_value": None,        # no sentinel fill value; only non-finite pixels are invalid
        "crop_bands": None,
    },
    "M3": {
        "input_channels": 84,
        "raw_channels": 84,        # slice.py already applied the 85 -> 84 crop
        "raw_root": "data/original - m3",
        "processed_root": "data/processed/M3",
        "packed_root": "data/packed/M3",
        "fill_value": -999.0,
        "crop_bands": 84,          # drop the last of 85 native bands
    },
    "AVIRIS": {
        "input_channels": 424,
        "raw_channels": 424,
        "raw_root": "data/original - AVIRIS",
        "processed_root": "data/processed/AVIRIS",
        "packed_root": "data/packed/AVIRIS",
        "fill_value": -9999.0,
        "crop_bands": None,
    },
    "CRIMS": {
        "input_channels": 456,
        "raw_channels": 457,       # on-disk band count; pack.py crops to 456
        "raw_root": "data/original - CRIMS",   # unused; CRIMS ships pre-processed
        "processed_root": "data/processed/CRIMS",
        "packed_root": "data/packed/CRIMS",
        "fill_value": None,
        "crop_bands": 456,         # drop the last of 457 bands (457 is prime)
    },
}


def make_settings(dataset: str) -> Settings:
    """
    Build a Settings instance whose input_channels (and therefore every
    derived spatial/spectral dim) matches the given dataset.

    Args:
        dataset : one of "IIRS", "M3", "AVIRIS", "CRIMS" (case-insensitive)

    Returns:
        Settings with input_channels overridden; all other fields keep the
        dataclass defaults.
    """
    key = dataset.upper()
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {sorted(DATASETS)}.")
    return Settings(input_channels=DATASETS[key]["input_channels"])


def probe_channels(dataset: str, processed_root: str | None = None):
    """
    Read the band count of one on-disk patch for ``dataset``.

    Looks at the packed shard first (``data/packed/<DS>/train.npy``, whose header
    carries the shape without reading the payload), then falls back to the first
    per-scene ``.npy`` under the processed root.

    Returns ``(count, source, location)`` where source is ``"packed"`` or
    ``"processed"``, or ``(None, None, None)`` if nothing is readable (no data
    staged yet — callers treat that as "cannot verify", not as an error).

    The SOURCE MATTERS and callers must honour it: a packed shard has already
    had ``crop_bands`` applied, so its count is the *effective* band count. A
    processed patch has not, so its count is the *raw* one. Conflating the two is
    what made every CRIMS slot fail — the probe returned the packed 456 and it
    was compared against ``raw_channels`` (457).
    """
    import numpy as np

    key = dataset.upper()
    meta = DATASETS[key]

    packed = Path(meta["packed_root"]) / "train.npy"
    if packed.is_file():
        try:
            return int(np.load(packed, mmap_mode="r").shape[-1]), "packed", str(packed)
        except (OSError, ValueError):
            pass

    root = Path(processed_root or meta["processed_root"])
    if not root.exists():
        return None, None, None
    for split in ("train", "valid", "test"):
        for p in root.glob(f"**/{split}/*.npy"):
            try:
                return int(np.load(p, mmap_mode="r").shape[-1]), "processed", str(p)
            except (OSError, ValueError):
                return None, None, None
    return None, None, None


def effective_channels(dataset: str, raw_c: int) -> int:
    """Band count the model actually sees: ``crop_bands`` if set, else ``raw_c``."""
    crop = DATASETS[dataset.upper()].get("crop_bands")
    return int(crop) if crop else int(raw_c)


def verify_channels(dataset: str, processed_root: str | None = None) -> None:
    """
    Raise a readable error if the on-disk band count disagrees with the config.

    Crop-aware: compares ``crop_bands`` (when set) against the raw on-disk count,
    and the post-crop count against ``input_channels``. Silently returns when no
    data is staged — there is nothing to check in that case.

    This exists because the failure it catches used to surface 500 frames deep as
    an opaque assertion in SpectralBranch.forward (CRIMS: config said 544, disk
    had 457).
    """
    key = dataset.upper()
    meta = DATASETS[key]
    count, source, where = probe_channels(key, processed_root)
    if count is None:
        return

    # A packed shard is ALREADY cropped, so its count is the effective band
    # count and must not be compared against raw_channels. Only the processed
    # tree carries the pre-crop count. Getting this wrong is what killed all
    # seven CRIMS slots: the probe read the packed 456 and it was checked
    # against raw_channels=457.
    if source == "processed":
        expected_raw = meta.get("raw_channels")
        if expected_raw is not None and count != expected_raw:
            raise ValueError(
                f"{key}: on-disk patches have {count} bands but DATASETS['{key}']"
                f"['raw_channels'] says {expected_raw}.\n"
                f"  probed        : {where}\n"
                f"Fix utils/config.py (and re-derive crop_bands / input_channels), "
                f"or point --data-root at the right directory.\n"
                f"Run `python utils/dataset/inspect_channels.py` for the full table."
            )
        eff = effective_channels(key, count)
    else:
        eff = count

    if eff != meta["input_channels"]:
        raise ValueError(
            f"{key}: effective band count is {eff} but input_channels says "
            f"{meta['input_channels']}.\n"
            f"  probed        : {where}  ({source})\n"
            f"  on-disk count : {count}"
            + (f"\n  crop_bands    : {meta.get('crop_bands')} (already applied "
               f"in the packed shard)" if source == "packed"
               else f"\n  crop_bands    : {meta.get('crop_bands')}")
            + "\nRun `python utils/dataset/inspect_channels.py` for the full table."
        )

    n_blocks = settings.spectral_n_1D_conv_blocks
    if eff % (2 ** n_blocks):
        raise ValueError(
            f"{key}: {eff} bands is not divisible by 2**spectral_n_1D_conv_blocks "
            f"= {2 ** n_blocks}, so the spectral encoder/decoder round-trip cannot "
            f"be exact. Set crop_bands to {eff - eff % (2 ** n_blocks)}."
        )


def apply_dataset(dataset: str, verify: bool = False,
                  processed_root: str | None = None) -> Settings:
    """
    Reconfigure the module-global ``settings`` in place for the given dataset.

    The branch modules (SpatialBranch / SpectralBranch) read attributes off the
    shared module-global ``settings`` object at forward time, so a single process
    can target any dataset's band count simply by mutating that global and
    recomputing the derived spatial/spectral dims.

    Call this once before building a model / dataloader for a given dataset.

    Args:
        dataset        : one of "IIRS", "M3", "AVIRIS", "CRIMS" (case-insensitive)
        verify         : cross-check the configured band count against what is
                         actually on disk, and raise early with a readable
                         message on mismatch.
        processed_root : override for the probe's search root (mirrors
                         --data-root); ignored when ``verify`` is False.

    Returns:
        The (mutated) module-global ``settings`` instance.
    """
    key = dataset.upper()
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {sorted(DATASETS)}.")
    settings.input_channels = DATASETS[key]["input_channels"]
    settings.__post_init__()   # recompute all derived spatial/spectral dims
    if verify:
        verify_channels(key, processed_root)
    return settings
