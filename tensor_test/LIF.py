import torch
from torch import nn
from ultralytics.nn.modules import Conv
import torch.nn.functional as F
from torchvision import transforms
from skimage import color

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
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x

class LP(nn.Module):
    """
    Light Prior Encoder，光照先验编码器

    输入:
        RGB可见光图像 x，shape为 [B, 3, H, W]
        数值范围应为 [0, 1]

    输出:
        光照权重图 W_illum，shape为 [B, 1, H/8, W/8]
        例如输入640×640，输出80×80
    """

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
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                f"LP需要输入[B,3,H,W]格式的RGB图像，"
                f"当前输入shape为{x.shape}"
            )

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

    def forward(self, F_vis, F_ir, W_illum):
        """
        Args:
            F_vis: [B,C,H,W]
            F_ir: [B,C,H,W]
            W_illum: [B,1,H,W]
        """
        # 光照门控
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
        F_out = F_base + F_illu
        return F_out

if __name__ == "__main__":
    # input_rgb = torch.rand(16, 3, 640, 640)
    #
    # model = LP()
    # weight = model(input_rgb)
    #
    # print("RGB shape:", input_rgb.shape)
    # print("RGB range:", input_rgb.min().item(), input_rgb.max().item())
    # print("Weight shape:", weight.shape)
    # print("Weight range:", weight.min().item(), weight.max().item())
    B, C, H, W = 16, 64, 80, 80
    F_vis = torch.randn(B, C, H, W)
    F_ir = torch.randn(B, C, H, W)
    W_illum = torch.rand(B, 1, H, W)

    fusion = ResidualIlluminationFusion(in_channels=C)
    F_out = fusion(F_vis, F_ir, W_illum)
    print(F_out.shape)  # [16, 64, 80, 80]