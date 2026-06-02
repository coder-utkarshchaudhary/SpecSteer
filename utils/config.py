from dataclasses import dataclass

@dataclass
class Settings:
    # Data
    input_height: int
    input_width: int
    input_channels: int

    # Spatial branch
    reduced_dims: int
    latent_dim: int
    n_2D_conv_blocks: int
    conv2D_kernel_size: int
    conv_output_c: int = 2**n_2D_conv_blocks*reduced_dims
    conv_output_h: int = input_height//2**n_2D_conv_blocks
    conv_output_w: int = input_width//2**n_2D_conv_blocks

    # Spectral branch
    spectral_n_1D_conv_blocks: int
    spectral_conv1D_kernel_size: int
    spectral_latent_dim: int
    spectral_linear_expansion_dim: int = 2*spectral_latent_dim
    spectral_transpose_c: int
    spectral_transpose_l: int

settings = Settings()