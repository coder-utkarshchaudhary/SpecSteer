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
    """

    # ------------------------------------------------------------------
    # Input patch shape (output of slice.py)
    # ------------------------------------------------------------------
    input_height: int = 64
    input_width: int = 64
    input_channels: int = 108          # bands 7:115 of the 256-band IIRS cube

    # ------------------------------------------------------------------
    # Preprocessing knobs (used by utils/dataset/preprocess.py)
    # ------------------------------------------------------------------
    band_start: int = 7                # inclusive band index in the raw 256-band cube
    band_end: int = 115               # exclusive
    # Index *within the selected 108-band subset* that corresponds to ≈1500 nm.
    # Absolute index 48 - band_start = 48 - 7 = 41 (matches file_processing.py).
    norm_band_idx: int = 41
    savgol_window: int = 7
    savgol_polyorder: int = 2

    # ------------------------------------------------------------------
    # Slicing knobs (used by utils/dataset/slice.py)
    # ------------------------------------------------------------------
    patch_size: int = 64
    patch_stride: int = 48            # 25% overlap: stride = patch_size * 0.75
    split_ratios: tuple = (0.70, 0.15, 0.15)   # (train, valid, test)

    # ------------------------------------------------------------------
    # Data paths
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
        )   # = 108 * 2 = 216

        # Encoder Conv1d final sequence length (halved each block):
        #   L_final = input_channels // 2^n
        self.spectral_transpose_l = self.input_channels // (
            2 ** self.spectral_n_1D_conv_blocks
        )   # = 108 // 4 = 27

        # Linear expansion size (what the decoder's linear layer must produce):
        self.spectral_linear_expansion_dim = (
            self.spectral_transpose_c * self.spectral_transpose_l
        )   # = 216 * 27 = 5832


settings = Settings()
