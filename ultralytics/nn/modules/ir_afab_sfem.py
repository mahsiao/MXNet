# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .block import C2f, C3k, C3k2
from .conv import Conv


class ConvolutionalGLU(nn.Module):
    """Convolutional gated linear unit with residual connection."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True, groups=hidden_features),
            act_layer(),
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_shortcut = x
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths per sample when applied in the main path of residual blocks."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Drop paths per sample."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AdaptiveMultiKernelConv2d(nn.Module):
    """Depthwise adaptive multi-kernel convolution."""

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        self.dwconv = nn.ModuleList(
            [
                nn.Conv2d(in_channels, in_channels, square_kernel_size, padding=square_kernel_size // 2, groups=in_channels),
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=(1, band_kernel_size),
                    padding=(0, band_kernel_size // 2),
                    groups=in_channels,
                ),
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=(band_kernel_size, 1),
                    padding=(band_kernel_size // 2, 0),
                    groups=in_channels,
                ),
            ]
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()
        self.dkw = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, in_channels * 3, 1))

    def forward(self, x):
        x_dkw = rearrange(self.dkw(x), "bs (g ch) h w -> g bs ch h w", g=3)
        x_dkw = F.softmax(x_dkw, dim=0)
        x = torch.stack([self.dwconv[i](x) * x_dkw[i] for i in range(len(self.dwconv))]).sum(0)
        return self.act(self.bn(x))


class AdaptiveMultiKernelDWConv(nn.Module):
    """Two-branch adaptive multi-kernel depthwise convolution."""

    def __init__(self, channel=256, kernels=(3, 5)):
        super().__init__()
        min_ch = channel // 2
        self.convs = nn.ModuleList(AdaptiveMultiKernelConv2d(min_ch, ks, ks * 3 + 2) for ks in kernels)
        self.conv_1x1 = Conv(channel, channel, k=1)

    def forward(self, x):
        _, c, _, _ = x.size()
        x_group = torch.split(x, [c // 2, c // 2], dim=1)
        x_group = torch.cat([self.convs[i](x_group[i]) for i in range(len(self.convs))], dim=1)
        return self.conv_1x1(x_group)


class AdaptiveMixerBlock(nn.Module):
    """Adaptive mixer block used by IR_AFAB."""

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mixer = AdaptiveMultiKernelDWConv(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = ConvolutionalGLU(dim)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class AFAB(C3k):
    """Adaptive feature aggregation block."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(AdaptiveMixerBlock(c_) for _ in range(n)))


class IR_AFAB(C3k2):
    """Infrared adaptive feature aggregation block."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(AFAB(self.c, self.c, 2, shortcut, g) if c3k else AdaptiveMixerBlock(self.c) for _ in range(n))


class ScharrConv(nn.Module):
    """Fixed Scharr edge convolution."""

    def __init__(self, channel):
        super().__init__()
        scharr_kernel_x = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=np.float32)
        scharr_kernel_y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=np.float32)

        scharr_kernel_x = torch.tensor(scharr_kernel_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        scharr_kernel_y = torch.tensor(scharr_kernel_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.scharr_kernel_x_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.scharr_kernel_y_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.scharr_kernel_x_conv.weight.data = scharr_kernel_x.expand(channel, 1, 3, 3).clone()
        self.scharr_kernel_y_conv.weight.data = scharr_kernel_y.expand(channel, 1, 3, 3).clone()
        self.scharr_kernel_x_conv.weight.requires_grad_(False)
        self.scharr_kernel_y_conv.weight.requires_grad_(False)

    def forward(self, x):
        grad_x = self.scharr_kernel_x_conv(x)
        grad_y = self.scharr_kernel_y_conv(x)
        return grad_x * 0.5 + grad_y * 0.5


class FreqSpatial(nn.Module):
    """Frequency-spatial feature extraction block."""

    def __init__(self, in_channels):
        super().__init__()
        self.sed = ScharrConv(in_channels)
        self.spatial_conv1 = Conv(in_channels, in_channels)
        self.spatial_conv2 = Conv(in_channels, in_channels)
        self.fft_conv = Conv(in_channels * 2, in_channels * 2, 3)
        self.fft_conv2 = Conv(in_channels, in_channels, 3)
        self.final_conv = Conv(in_channels, in_channels, 1)

    def forward(self, x):
        _, _, h, w = x.size()
        dtype = x.dtype

        spatial_feat = self.sed(x)
        spatial_feat = self.spatial_conv1(spatial_feat)
        spatial_feat = self.spatial_conv2(spatial_feat + x)

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_fft = x.float()
            fft_feat = torch.fft.rfft2(x_fft, norm="ortho")
            x_fft_real = torch.unsqueeze(torch.real(fft_feat), dim=-1)
            x_fft_imag = torch.unsqueeze(torch.imag(fft_feat), dim=-1)
            fft_feat = torch.cat((x_fft_real, x_fft_imag), dim=-1)
            fft_feat = rearrange(fft_feat, "b c h w d -> b (c d) h w").contiguous()
            fft_feat = fft_feat.to(self.fft_conv.conv.weight.dtype)
            fft_feat = self.fft_conv(fft_feat)
            fft_feat = rearrange(fft_feat, "b (c d) h w -> b c h w d", d=2).contiguous()
            fft_feat = fft_feat.float()
            fft_feat = torch.view_as_complex(fft_feat)
            fft_feat = torch.fft.irfft2(fft_feat, s=(h, w), norm="ortho")
            fft_feat = fft_feat.to(self.fft_conv2.conv.weight.dtype)
            fft_feat = self.fft_conv2(fft_feat)
        fft_feat = fft_feat.to(dtype)

        return self.final_conv(spatial_feat + fft_feat)


class SFEM(C2f):
    """Spatial-frequency enhancement module."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(FreqSpatial(self.c) for _ in range(n))
