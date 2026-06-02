from torch import nn
from torch.nn import Conv1d, Conv2d

from utils.config import settings

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1D_block = Conv1d(
            in_channels=1,
            out_channels=settings.reduced_dims,
            kernel_size=settings.input_channels
        )
        
        in_c = settings.reduced_dims
        out_c = 2*in_c

        conv2d_layers = []
        for _ in range(settings.n_2D_conv_blocks):
            layer = Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=settings.conv2D_kernel_size,
                stride=2,
                padding=1
            )
            conv2d_layers.append(layer)
            conv2d_layers.append(nn.ReLU())
            in_c=out_c
            out_c*=2
        
        self.conv2D_block = nn.Sequential(*conv2d_layers)

        self.flatten = nn.Flatten()

        self.linear = nn.LazyLinear(settings.latent_dim)

    def forward(self, x):
        """
            x: (B, H, W, C)
            returns:
                (B, 256)
        """
        batch, h, w, c = x.shape
        assert h==settings.input_height, f"SPATIAL ENCODER: Mismatch in height. Expected: {settings.input_height} found: {h}."
        assert w==settings.input_width, f"SPATIAL ENCODER: Mismatch in width. Expected: {settings.input_width} found: {w}."
        assert c==settings.input_channels,  f"SPATIAL ENCODER: Mismatch in channels. Expected: {settings.input_channels} found: {c}."

        x = x.reshape(batch * h * w, 1, c)
        
        x = self.conv1D_block(x)
        # (B*H*W, reduced_dim, 1)

        x = x.squeeze(-1)
        # (B*H*W, reduced_dim)

        x = x.reshape(batch, h, w, settings.reduced_dims)

        x = x.permute(0, 3, 1, 2)
        # (B, reduced_dim, H, W)

        x = self.conv2D_block(x)
        # (B, 2**settings.n_2D_conv_blocks*settings.reduced_dim, settings.input_height/2**settings.n_2D_conv_blocks, settings.input_width/2**settings.n_2D_conv_blocks)
        
        x = self.flatten(x)
        x = self.linear(x)
        # (B, 256)

        return x

class Decoder(nn.Module):
    def __init__(self, conv_output_c, conv_output_h, conv_output_w):
        super().__init__()
        self.conv_out_c = conv_output_c
        self.conv_out_h = conv_output_h
        self.conv_out_w = conv_output_w

        flatten = conv_output_c*conv_output_h*conv_output_w

        self.linear = nn.Linear(
            in_features=256,
            out_features=flatten
        )

        in_c = conv_output_c

        layers = []
        for _ in range(settings.n_2D_conv_blocks):
            out_c = in_c // 2
            layer = nn.ConvTranspose2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            
            layers.append(layer)
            layers.append(nn.ReLU())

            in_c = out_c
        
        self.transposeconv2D_block = nn.Sequential(*layers)

        self.conv1D_block = Conv1d(
            in_channels=1,
            out_channels=settings.input_channels,
            kernel_size=settings.reduced_dims
        )
    
    def forward(self, z):
        """
        z: (B, 256)
        returns:
            (B, H, W, Bands)
        """

        B = z.shape[0]

        x = self.linear(z)

        x = x.view(
            B,
            self.conv_out_c,
            self.conv_out_h,
            self.conv_out_w,
        )

        x = self.transposeconv2D_block(x)
        # (B, reduced_dim, H, W)
        
        assert x.shape[1] == settings.reduced_dims, f"SPATIAL DECODER: Mismatch in channels. Expected reduced_dim={settings.reduced_dims}, found {x.shape[1]}"
        assert x.shape[2] == settings.input_height, f"SPATIAL DECODER: Mismatch in height. Expected height={settings.input_height}, found {x.shape[2]}"
        assert x.shape[3] == settings.input_width, f"SPATIAL DECODER: Mismatch in width. Expected width={settings.input_width}, found {x.shape[3]}"

        H, W = x.shape[2], x.shape[3]

        x = x.permute(0, 2, 3, 1)
        # (B, H, W, reduced_dim)

        x = x.reshape(
            B * H * W,
            1,
            settings.reduced_dims
        )
        
        x = self.conv1D_block(x)
        # (B*H*W, Bands, 1)

        x = x.squeeze(-1)
        # (B*H*W, Bands)

        x = x.reshape(B, settings.input_height, settings.input_width, settings.input_channels)
        # (B, H, W, C)
        
        return x

class SpatialEncoderDecoder(nn.Module):
    def __init__(self, conv_output_c, conv_output_h, conv_output_w,):
        super().__init__()

        self.encoder = Encoder()

        self.decoder = Decoder(
            conv_output_c=conv_output_c,
            conv_output_h=conv_output_h,
            conv_output_w=conv_output_w,
        )

    def forward(self, x):
        """
        x: (B, H, W, C)
        returns:
            latent: (B, latent_dim)
            reconstruction: (B, H, W, C)
        """

        latent = self.encoder(x)
        reconstruction = self.decoder(latent)

        return latent, reconstruction