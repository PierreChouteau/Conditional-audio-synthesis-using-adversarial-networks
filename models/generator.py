import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import calculate_gain, kaiming_normal_


class Reshape(torch.nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)


class PixelNormalization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = x / (1 / x.size(1) * (x**2).sum(dim=1, keepdim=True) + 1e-8).sqrt()
        return out


class UpsampleLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        k_filters,
        k_width,
        k_heigth,
        padding="same",
        scale_factor=2,
        upsample_mode="nearest",
        bias=True,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode=upsample_mode),
            nn.Conv2d(
                in_dim,
                k_filters,
                kernel_size=(k_width, k_heigth),
                padding=padding,
                bias=bias,
            ),
            nn.LeakyReLU(0.2),
            PixelNormalization(),
            
            nn.Conv2d(
                k_filters,
                k_filters,
                kernel_size=(k_width, k_heigth),
                padding=padding,
                bias=bias,
            ),
            nn.LeakyReLU(0.2),
            PixelNormalization(),
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim=256,
        out_channels=1,
        k_width=3,
        k_heigth=3,
        k_filters=32,
        padding="same",
        scale_factor=2,
        upsample_mode="nearest",
    ):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.k_width = k_width
        self.k_height = k_heigth
        self.k_filters = k_filters
        self.upsample_mode = upsample_mode
        self.scale_factor = scale_factor
        self.input_dim = self.latent_dim

        self.model = nn.Sequential(
            nn.Conv2d(
                self.k_filters * (2**3),
                self.k_filters * (2**3),
                kernel_size=(2, 16),
                padding=(1, 15),
                bias=False,
            ),
            nn.LeakyReLU(0.2),
            PixelNormalization(),
            
            nn.Conv2d(
                self.k_filters * (2**3),
                self.k_filters * (2**3),
                kernel_size=(self.k_width, self.k_height),
                padding=padding,
                bias=False,
            ),
            nn.LeakyReLU(0.2),
            PixelNormalization(),
            
            UpsampleLayer(
                self.k_filters * (2**3),
                self.k_filters * (2**3),
                self.k_width,
                self.k_height,
                padding=padding,
                scale_factor=self.scale_factor,
                upsample_mode=self.upsample_mode,
                bias=False,
            ),
            UpsampleLayer(
                self.k_filters * (2**3),
                self.k_filters * (2**3),
                self.k_width,
                self.k_height,
                padding=padding,
                scale_factor=self.scale_factor,
                upsample_mode=self.upsample_mode,
                bias=False,
            ),
            UpsampleLayer(
                self.k_filters * (2**3),
                self.k_filters * (2**3),
                self.k_width,
                self.k_height,
                padding=padding,
                scale_factor=self.scale_factor,
                upsample_mode=self.upsample_mode,
                bias=False,
            ),
            UpsampleLayer(
                self.k_filters * (2**3),
                self.k_filters * (2**2),
                self.k_width,
                self.k_height,
                padding=padding,
                scale_factor=self.scale_factor,
                upsample_mode=self.upsample_mode,
                bias=False,
            ),
            UpsampleLayer(
                self.k_filters * (2**2),
                self.k_filters * 2,
                self.k_width,
                self.k_height,
                padding=padding,
                scale_factor=self.scale_factor,
                upsample_mode=self.upsample_mode,
                bias=False,
            ),
            UpsampleLayer(
                self.k_filters * 2,
                self.k_filters,
                self.k_width,
                self.k_height,
                padding=padding,
                scale_factor=self.scale_factor,
                upsample_mode=self.upsample_mode,
                bias=False,
            ),
            
            nn.Conv2d(
                self.k_filters,
                self.out_channels,
                kernel_size=(1, 1),
                padding=padding,
                bias=False,
            ),
            nn.Tanh(),
        )

    def initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d)):
                kaiming_normal_(m.weight, a=calculate_gain("conv2d"))

    def forward(self, x):
        x = x.view(-1, x.size(-1), 1, 1)
        output = self.model(x)
        return output


###########################################################
# Just a test function to verify the output generator size
###########################################################
def test_gen(TEST=False):
    if TEST:
        noise = torch.randn(8, 256)
        gen = Generator()

        image_test = gen(noise)
        print(noise.size(), image_test.size())

test_gen(False)