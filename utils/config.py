from dataclasses import dataclass, field


@dataclass
class Settings:
    """
    Centralised configuration for the Dual-Stream Physics-Informed VAE pipeline.

    All derived / computed fields are populated in __post_init__, so only the
    primary fields need to be set when instantiating Settings().

    Spectral arithmetic (with spectral_conv1D_kernel_size=4, stride=2, padding=1):
      Conv1d:          L_out = L_in // 2
      ConvTranspose1d: L_out = 2 * L_in
    With n=2 blocks and input_channels=108:
      Encoder:  108 → 54 → 27  (L);  channels  1 → 108 → 216
      Decoder:   27 → 54 → 108 (L);  channels 216 → 108 → 1

    NOTE: input_channels varies per dataset (IIRS=256, M3=84, AVIRIS=424) — see
    DATASETS / make_settings() below. Every dataset needs its own Settings
    instance (and therefore its own model instance) since band count differs.
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

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    batch_size: int = 32
    num_workers: int = 4

    # ------------------------------------------------------------------
    # Spatial branch
    # ------------------------------------------------------------------
    reduced_dims: int = 32            # Conv1D channel reduction per pixel spectrum
    latent_dim: int = 256             # spatial latent dim (after reparameterize chunk)
    n_2D_conv_blocks: int = 4        # 64 / 2^4 = 4 px spatial bottleneck
    conv2D_kernel_size: int = 3

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

    # Derived spectral (computed in __post_init__)
    spectral_linear_expansion_dim: int = field(init=False)
    spectral_transpose_c: int = field(init=False)
    spectral_transpose_l: int = field(init=False)

    # ------------------------------------------------------------------
    # Baseline capacity knobs (overridden per-dataset by hyperparam YAML
    # so each baseline matches vae-our's param count at that dataset).
    # Defaults here are the IIRS picks.
    # ------------------------------------------------------------------
    vae_standard_base_ch: int = 134
    vae_standard_n_down: int = 3
    vae_standard_latent_ch: int = 16

    vae_3d_base_ch: int = 78
    vae_3d_n_down: int = 3
    vae_3d_latent_ch: int = 8

    vae_1d_hidden_dims: tuple = (4224, 2112, 1056)
    vae_1d_latent_dim: int = 32

    def __post_init__(self):
        # ---- Spatial derived fields ----
        self.conv_output_c = self.reduced_dims * (2 ** self.n_2D_conv_blocks)
        self.conv_output_h = self.input_height // (2 ** self.n_2D_conv_blocks)
        self.conv_output_w = self.input_width // (2 ** self.n_2D_conv_blocks)

        # ---- Spectral derived fields ----
        # Encoder Conv1d final channel count:
        #   Block i goes from (in_c → out_c) where out_c doubles each step starting at input_channels.
        #   After n blocks: final channel = input_channels * 2^(n-1)
        self.spectral_transpose_c = self.input_channels * (
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
DATASETS = {
    "IIRS": {
        "input_channels": 256,
        "raw_root": "data/original - IIRS",
        "processed_root": "data/processed/IIRS",
        "fill_value": None,        # no sentinel fill value; only non-finite pixels are invalid
        "crop_bands": None,
    },
    "M3": {
        "input_channels": 84,
        "raw_root": "data/original - m3",
        "processed_root": "data/processed/M3",
        "fill_value": -999.0,
        "crop_bands": 84,          # drop the last of 85 native bands
    },
    "AVIRIS": {
        "input_channels": 424,
        "raw_root": "data/original - AVIRIS",
        "processed_root": "data/processed/AVIRIS",
        "fill_value": -9999.0,
        "crop_bands": None,
    },
}


def make_settings(dataset: str) -> Settings:
    """
    Build a Settings instance whose input_channels (and therefore every
    derived spatial/spectral dim) matches the given dataset.

    Args:
        dataset : one of "IIRS", "M3", "AVIRIS" (case-insensitive)

    Returns:
        Settings with input_channels overridden; all other fields keep the
        dataclass defaults.
    """
    key = dataset.upper()
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {sorted(DATASETS)}.")
    return Settings(input_channels=DATASETS[key]["input_channels"])


def apply_dataset(dataset: str) -> Settings:
    """
    Reconfigure the module-global ``settings`` in place for the given dataset.

    The branch modules (SpatialBranch / SpectralBranch) read attributes off the
    shared module-global ``settings`` object at forward time, so a single process
    can target any dataset's band count simply by mutating that global and
    recomputing the derived spatial/spectral dims.

    Call this once before building a model / dataloader for a given dataset.

    Args:
        dataset : one of "IIRS", "M3", "AVIRIS" (case-insensitive)

    Returns:
        The (mutated) module-global ``settings`` instance.
    """
    key = dataset.upper()
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {sorted(DATASETS)}.")
    settings.input_channels = DATASETS[key]["input_channels"]
    settings.__post_init__()   # recompute all derived spatial/spectral dims
    return settings
