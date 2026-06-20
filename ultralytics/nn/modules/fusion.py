import torch
from torch import nn
import torch.nn.functional as F
from .conv import Conv
"""
Light Prior Encoder，光照先验编码器

输入:
    RGB可见光图像 x，shape为 [B, 3, H, W]
    数值范围应为 [0, 1]

输出:
    光照权重图 W_illum，shape为 [B, 1, H/8, W/8]
    例如输入640×640，输出80×80
"""
class LP(nn.Module):

    def __init__(self):
        super().__init__()

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 输出单通道权重，不接BN和ReLU
        self.conv4 = nn.Conv2d(
            64,
            1,
            kernel_size=1,
            padding=0
        )

    @staticmethod
    def rgb_to_luminance(x):
        """
        使用BT.709感知亮度公式近似提取亮度通道。

        Args:
            x: RGB图像，[B,3,H,W]，数值范围[0,1]

        Returns:
            L: 亮度先验，[B,1,H,W]，数值范围[0,1]
        """
        if x.ndim != 4 or x.shape[1] not in (3, 4):
            raise ValueError(
                f"LP需要输入[B,3,H,W]格式的RGB图像，"
                f"当前输入shape为{x.shape}"
            )

        # Support RGBT input by using only the visible RGB channels.
        x = x[:, :3, :, :]
        r = x[:, 0:1, :, :]
        g = x[:, 1:2, :, :]
        b = x[:, 2:3, :, :]

        L = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return L

    def forward(self, x):
        """
        Args:
            x: RGB图像，[B,3,H,W]，范围[0,1]

        Returns:
            W_illum: 光照权重图，[B,1,H/8,W/8]
        """
        # RGB → 近似亮度通道
        L = self.rgb_to_luminance(x)

        # 640 → 320
        x = self.pool(L)
        x = self.conv1(x)

        # 320 → 160
        x = self.pool(x)
        x = self.conv2(x)

        # 160 → 80
        x = self.pool(x)
        x = self.conv3(x)

        # 映射为单通道光照权重
        x = self.conv4(x)
        W_illum = torch.sigmoid(x)

        return W_illum


class ResidualIlluminationFusion(nn.Module):
    """
    光照感知残差融合模块
    输入:
        F_vis: 可见光特征 [B, C, H, W]
        F_ir: 红外特征 [B, C, H, W]
        W_illum: 光照权重 [B,1,H,W]
    输出:
        F_out: 融合后特征 [B, C, H, W]
    """

    def __init__(self, in_channels):
        super().__init__()
        # 光照引导融合分支
        self.fuse_conv1 = nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=1, padding=0)
        self.fuse_act = nn.SiLU(inplace=True)
        self.fuse_conv3 = nn.Conv2d(2*in_channels, in_channels, kernel_size=3, padding=1)

        # 基础融合分支
        self.base_conv = nn.Conv2d(2*in_channels, in_channels, kernel_size=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Args:
            F_vis: [B,C,H,W]
            F_ir: [B,C,H,W]
            W_illum: [B,1,H,W]
        """
        # 光照门控
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError(
                "ResidualIlluminationFusion expects [F_vis, F_ir, W_illum] as input, "
                f"but got {type(x)} with length {len(x) if isinstance(x, (list, tuple)) else 'N/A'}"
            )

        F_vis, F_ir, W_illum = x

        if F_vis.shape != F_ir.shape:
            raise ValueError(
                "ResidualIlluminationFusion expects F_vis and F_ir to have the same shape, "
                f"but got {F_vis.shape} and {F_ir.shape}"
            )

        if W_illum.ndim != 4 or W_illum.shape[1] != 1:
            raise ValueError(
                "ResidualIlluminationFusion expects W_illum with shape [B,1,H,W], "
                f"but got {W_illum.shape}"
            )

        if W_illum.shape[-2:] != F_vis.shape[-2:]:
            W_illum = F.interpolate(W_illum, size=F_vis.shape[-2:], mode="bilinear", align_corners=False)

        W_illum = W_illum.to(dtype=F_vis.dtype, device=F_vis.device)

        F_vis_g = W_illum * F_vis
        F_ir_g = (1 - W_illum) * F_ir

        # 拼接卷积重构
        F_cat = torch.cat([F_vis_g, F_ir_g], dim=1)
        F_illu = self.fuse_conv1(F_cat)
        F_illu = self.fuse_act(F_illu)
        F_illu = self.fuse_conv3(F_illu)

        # 基础残差融合分支
        F_base = self.base_conv(torch.cat([F_vis, F_ir], dim=1))

        # 残差融合
        F_out = F_base + self.gamma * F_illu
        return F_out

###LIFAdd
class LIF(nn.Module):
    # forward返回一个80*80的权重图
    def __init__(self):
        super(LIF, self).__init__()
        # 640*640 -> 80*80
        self.conv1 = Conv(3, 32, k=3, p=1)
        self.conv2 = Conv(32, 64, k=3, p=1)
        self.conv3 = Conv(64, 64, k=3, p=1)
        self.conv4 = Conv(64, 1, k=1, p=0)
        self.relu = nn.ReLU()

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] not in (3, 4):
            raise ValueError(f"LIF expects RGB/RGBT input with shape [B,3/4,H,W], got {x.shape}.")
        x = x[:, :3, :, :]
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x


class LIFAdd(nn.Module):
    def __init__(self, layer):
        super(LIFAdd, self).__init__()
        self.layer = layer
        self.pool_layer4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool_layer5 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.beta = 0.4

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError(
                "LIFAdd expects [F_ir, F_rgb, weight] as input, "
                f"but got {type(x)} with length {len(x) if isinstance(x, (list, tuple)) else 'N/A'}"
            )
        x_ir = x[0]
        x_rgb = x[1]
        weight = x[2]
        if x_ir.shape != x_rgb.shape:
            raise ValueError(f"LIFAdd expects equal IR/RGB feature shapes, got {x_ir.shape} and {x_rgb.shape}.")
        if weight.ndim != 4 or weight.shape[1] != 1:
            raise ValueError(f"LIFAdd expects a single-channel weight map [B,1,H,W], got {weight.shape}.")

        step1 = (weight - 0.31) / 0.63

        step2 = torch.clamp(step1, max=0.5)

        weight = self.beta * step2 + 0.5
        if self.layer == 3:
            pass
        elif self.layer == 4:
            weight = self.pool_layer4(weight)
        elif self.layer == 5:
            weight = self.pool_layer5(weight)
        if weight.shape[-2:] != x_rgb.shape[-2:]:
            weight = F.interpolate(weight, size=x_rgb.shape[-2:], mode="bilinear", align_corners=False)
        weight = weight.to(dtype=x_rgb.dtype, device=x_rgb.device)
        # h, w = x_rgb.shape[2:]
        # print(h, w)
        weight_rgb = weight
        weight_ir = 1 - weight
        return weight_rgb * x_rgb + weight_ir * x_ir

######2025-08 add CIFusion###########################################
class CIFusion(nn.Module):
    # Concat the rgb and infrared feature maps
    def __init__(self, c1, r=16, dimension=1):  # c1 is the channel count of 1 input stream
        super().__init__()
        self.c1_single_stream = c1  # Store single stream channel count
        self.c_total_concat = c1 * 2  # Total channels after concatenation
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.c_total_concat, self.c_total_concat // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.c_total_concat // r, self.c_total_concat, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):  # x is the concatenated feature map [RGB_features, IR_features]
        b, _, _, _ = x.size()  # b, c_total_concat, h, w
        y = self.avg_pool(x).view(b, self.c_total_concat)
        y = self.fc(y).view(b, self.c_total_concat, 1, 1)

        x1 = x * y

        return x + torch.cat((x1[:, self.c1_single_stream:, ...], x1[:, :self.c1_single_stream, ...]), dim=1)


#####2025-08 add CIFusionV2########################
class CIFusion_v2(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, r=16, dimension=1):
        super().__init__()

    def forward(self, x):
        return x


######2025-08 add CIFusionV3##########
class CIFusion_v3(nn.Module):
    """
    CIFusion_v3: CIFusion module WITHOUT the channel attention mechanism.
                 'y' (attention weights) is effectively always 1.
                 The unique cross-attention summation remains.
                 This helps evaluate the contribution of channel attention.
    """

    def __init__(self, c1, r=16, dimension=1):
        super().__init__()
        self.c1_single_stream = c1
        self.c_total_concat = c1 * 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.c_total_concat, self.c_total_concat // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.c_total_concat // r, self.c_total_concat, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = x
        return x + torch.cat((x1[:, self.c1_single_stream:, ...], x1[:, :self.c1_single_stream, ...]), dim=1)


######2025-08 add CIFusionV4##########
class CIFusion_v4(nn.Module):
    """
    CIFusion_v4: CIFusion module with channel attention,
        but replaces the unique cross-attention summation
        with simply returning the attention-gated features (x * y).
        This evaluates the contribution of the unique summation.
    """

    def __init__(self, c1, r=16, dimension=1):
        super().__init__()
        self.c1_single_stream = c1
        self.c_total_concat = c1 * 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.c_total_concat, self.c_total_concat // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.c_total_concat // r, self.c_total_concat, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, self.c_total_concat)
        y = self.fc(y).view(b, self.c_total_concat, 1, 1)

        x1 = x * y

        return x1


##########2025-08 add CIFusionV5##########
class BasicSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)

        attention_map = self.sigmoid(self.conv1(x_cat))
        return x * attention_map


class CIFusion_v5(nn.Module):
    def __init__(self, c1, r=16, dimension=1):
        super().__init__()
        self.c1_single_stream = c1
        self.c_total_concat = c1 * 2
        self.attention_module = BasicSpatialAttention(kernel_size=7)

    def forward(self, x):
        x1 = self.attention_module(x)

        return x + torch.cat((x1[:, self.c1_single_stream:, ...], x1[:, :self.c1_single_stream, ...]), dim=1)


#### 2025-08 CIFusionV6 ##########

# Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction_ratio, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.mlp(self.max_pool(x).squeeze(-1).squeeze(-1))
        scale = self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return x * scale


# Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv1(x_cat))
        return x * attention_map


# CBAM Block
class CBAM(nn.Module):
    def __init__(self, channel, reduction_ratio=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channel, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)  # channel attention
        x = self.spatial_attention(x)  # spatial attention
        return x


class CIFusion_v6(nn.Module):
    def __init__(self, c1, r=16, dimension=1):
        super().__init__()
        self.c1_single_stream = c1
        self.c_total_concat = c1 * 2
        self.attention_module = CBAM(channel=self.c_total_concat, reduction_ratio=r)

    def forward(self, x):
        x1 = self.attention_module(x)

        return x + torch.cat((x1[:, self.c1_single_stream:, ...], x1[:, :self.c1_single_stream, ...]), dim=1)


#######2025-08 ADD block############
class ADD(nn.Module):
    def __init__(self, arg):
        super(ADD, self).__init__()
        self.arg = arg

    def forward(self, x):
        return torch.add(x[0], x[1])
