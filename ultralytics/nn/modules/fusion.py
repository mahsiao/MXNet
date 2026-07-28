import torch
from torch import nn
import torch.nn.functional as F
from .conv import Conv


class FCFS(nn.Module):
    """
    Frequency-domain Coherence-based Feature Separation for RGB-IR features.

    This module only separates paired RGB/IR features into shared, RGB-private,
    and IR-private features. It does not perform feature fusion.

    Usage:
        shared, rgb_private, ir_private = FCFS()([rgb_p3, ir_p3])

    In YOLO YAML, use one FCFS layer for one feature level. For example, P3, P4
    and P5 should call this module separately, just like other single-scale
    fusion modules.
    """

    def __init__(
        self,
        eps=1e-6,
        tau=0.25,
        temp=0.2,
        share_scale=0.7,
        private_boost=1.5,
    ):
        super().__init__()
        self.eps = eps
        self.tau = tau
        self.temp = temp
        self.share_scale = share_scale
        self.private_boost = private_boost

    def forward(self, x):
        """
        Args:
            x: [F_rgb, F_ir], each with shape [B,C,H,W].

        Returns:
            [F_shared, F_rgb_private, F_ir_private]
        """
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(
                "FCFS expects [F_rgb, F_ir], "
                f"but got {type(x)} with length {len(x) if isinstance(x, (list, tuple)) else 'N/A'}."
            )

        rgb_feats, ir_feats = x
        F_shared, F_rgb_private, F_ir_private = self.freq_coherence_split(rgb_feats, ir_feats)
        return [F_shared, F_rgb_private, F_ir_private]

    def freq_coherence_split(self, x, y):
        """
        Args:
            x: RGB feature map, [B,C,H,W].
            y: IR feature map, [B,C,H,W].

        Returns:
            Zc: shared feature.
            Zr: RGB-private feature.
            Zi: IR-private feature.
        """
        if x.shape != y.shape:
            raise ValueError(f"FCFS expects paired RGB/IR features with the same shape, got {x.shape} and {y.shape}.")
        if x.ndim != 4:
            raise ValueError(f"FCFS expects feature maps with shape [B,C,H,W], got {x.shape}.")

        dtype = x.dtype
        spatial_size = x.shape[-2:]

        # FFT runs in float32 for numerical stability, especially under AMP.
        with torch.cuda.amp.autocast(enabled=False):
            x_float = x.float()
            y_float = y.float()

            X = torch.fft.rfft2(x_float, norm="ortho")
            Y = torch.fft.rfft2(y_float, norm="ortho")

            Sxx = (X.real ** 2 + X.imag ** 2) + self.eps
            Syy = (Y.real ** 2 + Y.imag ** 2) + self.eps
            Sxy = X * torch.conj(Y)

            # Spectral coherence estimates cross-modal shared responses.
            coh = (Sxy.real ** 2 + Sxy.imag ** 2) / (Sxx * Syy)

            # Balance term prevents the stronger modality from dominating.
            strength_balance = torch.sqrt(Sxx * Syy) / (Sxx + Syy + self.eps)

            Mc = torch.sigmoid(((coh * strength_balance) - self.tau) / self.temp)
            Mc = Mc * self.share_scale + (1.0 - self.share_scale) * 0.5
            Mc = Mc.clamp(0.0, 1.0)

            diff_mag = torch.abs(Sxx - Syy) / (Sxx + Syy + self.eps)
            Mr = (1.0 - Mc) * diff_mag * (Sxx / (Sxx + Syy))
            Mi = (1.0 - Mc) * diff_mag * (Syy / (Sxx + Syy))

            Mr = Mr * self.private_boost
            Mi = Mi * self.private_boost

            denom = Mc + Mr + Mi + self.eps
            Mc = Mc / denom
            Mr = Mr / denom
            Mi = Mi / denom

            Zc_hat = Mc * (X + Y) * 0.5
            Zr_hat = Mr * X
            Zi_hat = Mi * Y

            Zc = torch.fft.irfft2(Zc_hat, s=spatial_size, norm="ortho")
            Zr = torch.fft.irfft2(Zr_hat, s=spatial_size, norm="ortho")
            Zi = torch.fft.irfft2(Zi_hat, s=spatial_size, norm="ortho")

        return Zc.to(dtype), Zr.to(dtype), Zi.to(dtype)



class FCFSShared(FCFS):
    """
    FCFS shared-only variant for ablation experiments.

    Input:
        [F_rgb, F_ir], where both tensors have shape [B,C,H,W].

    Output:
        F_shared with shape [B,C,H,W].

    This module keeps only the coherence-based shared feature and discards
    RGB-private/IR-private features. No extra Conv is used here, so the ablation
    directly reflects the contribution of the FCFS shared branch.
    """

    def forward(self, x):
        F_shared, _, _ = super().forward(x)
        return F_shared


class FCFSPrivateResidual(FCFS):
    """
    FCFS shared + private residual variant for ablation experiments.

    Input:
        [F_rgb, F_ir], where both tensors have shape [B,C,H,W].

    Output:
        F_out with shape [B,C,H,W].

    The module first separates shared/RGB-private/IR-private features using
    FCFS, then compresses the two private branches back to C channels and adds
    them to the shared feature with a learnable residual scale.
    """

    def __init__(
        self,
        c1,
        eps=1e-6,
        tau=0.25,
        temp=0.2,
        share_scale=0.7,
        private_boost=1.5,
    ):
        super().__init__(
            eps=eps,
            tau=tau,
            temp=temp,
            share_scale=share_scale,
            private_boost=private_boost,
        )
        self.private_conv = nn.Sequential(
            nn.Conv2d(2 * c1, c1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        F_shared, F_rgb_private, F_ir_private = super().forward(x)
        F_private = self.private_conv(torch.cat([F_rgb_private, F_ir_private], dim=1))
        return F_shared + self.gamma * F_private


class SPAF(FCFS):
    """
    Shared-Private Adaptive Fusion for YOLO RGB-IR detection.

    This module is inspired by the TriAdapt idea in SPFD, but is redesigned for
    CNN-based YOLO feature maps. It keeps the core principle of using the shared
    feature as semantic Query and modality-private features as Key/Value, then
    explicitly fuses shared and private representations instead of treating the
    private branch as a weak residual only.

    Inputs:
        [F_rgb, F_ir] or [F_rgb, F_ir, W_illum].
        F_rgb/F_ir: [B,C,H,W]. W_illum is optional [B,1,h,w].

    Output:
        F_out: [B,C,H,W].
    """

    def __init__(
        self,
        c1,
        num_heads=4,
        kv_stride=2,
        attn_ratio=1.0,
        dropout=0.0,
        gamma_init=0.1,
        eps=1e-6,
        tau=0.25,
        temp=0.2,
        share_scale=0.7,
        private_boost=1.5,
    ):
        super().__init__(
            eps=eps,
            tau=tau,
            temp=temp,
            share_scale=share_scale,
            private_boost=private_boost,
        )
        hidden_dim = max(int(c1 * attn_ratio), num_heads)
        if hidden_dim % num_heads != 0:
            hidden_dim = ((hidden_dim + num_heads - 1) // num_heads) * num_heads

        self.c1 = c1
        self.hidden_dim = hidden_dim
        self.kv_stride = max(int(kv_stride), 1)

        self.illum_proj = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.q_proj = nn.Conv2d(c1, hidden_dim, kernel_size=1, bias=False)
        self.rgb_kv_proj = nn.Conv2d(c1, hidden_dim, kernel_size=1, bias=False)
        self.ir_kv_proj = nn.Conv2d(c1, hidden_dim, kernel_size=1, bias=False)

        self.q_norm = nn.LayerNorm(hidden_dim)
        self.rgb_norm = nn.LayerNorm(hidden_dim)
        self.ir_norm = nn.LayerNorm(hidden_dim)

        self.cross_rgb = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_ir = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.rgb_enhance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ir_enhance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.private_out = nn.Sequential(
            nn.Conv2d(hidden_dim, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2 * c1, c1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.gamma = nn.Parameter(torch.ones(1) * float(gamma_init))

    @staticmethod
    def _to_tokens(feat):
        """Convert [B,C,H,W] to [B,HW,C] tokens."""
        return feat.flatten(2).transpose(1, 2).contiguous()

    def _pool_kv(self, feat):
        """Downsample private K/V features to keep P3 attention lightweight."""
        if self.kv_stride <= 1:
            return feat
        return F.avg_pool2d(feat, kernel_size=self.kv_stride, stride=self.kv_stride, ceil_mode=True)

    def _make_query_source(self, F_shared, W_illum=None):
        """Inject optional illumination prior into the shared query source."""
        if W_illum is None:
            return F_shared
        if W_illum.ndim != 4 or W_illum.shape[1] != 1:
            raise ValueError(f"SPAF expects W_illum with shape [B,1,H,W], got {W_illum.shape}.")
        if W_illum.shape[-2:] != F_shared.shape[-2:]:
            W_illum = F.interpolate(W_illum, size=F_shared.shape[-2:], mode="bilinear", align_corners=False)
        return F_shared + self.illum_proj(W_illum.to(dtype=F_shared.dtype))

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) not in (2, 3):
            raise ValueError(
                "SPAF expects [F_rgb, F_ir] or [F_rgb, F_ir, W_illum], "
                f"but got {type(x)} with length {len(x) if isinstance(x, (list, tuple)) else 'N/A'}."
            )

        F_rgb, F_ir = x[0], x[1]
        W_illum = x[2] if len(x) == 3 else None
        F_shared, F_rgb_private, F_ir_private = self.freq_coherence_split(F_rgb, F_ir)
        b, _, h, w = F_shared.shape

        q_source = self._make_query_source(F_shared, W_illum)
        q = self._to_tokens(self.q_proj(q_source))
        rgb_kv = self._to_tokens(self.rgb_kv_proj(self._pool_kv(F_rgb_private)))
        ir_kv = self._to_tokens(self.ir_kv_proj(self._pool_kv(F_ir_private)))

        q = self.q_norm(q)
        rgb_kv = self.rgb_norm(rgb_kv)
        ir_kv = self.ir_norm(ir_kv)

        cross_rgb, _ = self.cross_rgb(q, rgb_kv, rgb_kv, need_weights=False)
        cross_ir, _ = self.cross_ir(q, ir_kv, ir_kv, need_weights=False)

        rgb_private_tokens = self.rgb_enhance(cross_rgb)
        ir_private_tokens = self.ir_enhance(cross_ir)
        gate_input = torch.cat([q, cross_rgb, cross_ir], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        private_tokens = gate * rgb_private_tokens + (1.0 - gate) * ir_private_tokens

        private_feat = private_tokens.transpose(1, 2).reshape(b, self.hidden_dim, h, w)
        private_feat = self.private_out(private_feat)

        fused = self.fusion_conv(torch.cat([F_shared, private_feat], dim=1))
        return fused + self.gamma * private_feat


class FCFSTriAdapt(SPAF):
    """Backward-compatible name for SPAF in existing YAML files."""

    pass


class ISPF(FCFS):
    """
    Illumination-guided Shared-Private Fusion for P3 RGB-IR features.

    Input:
        [F_rgb, F_ir, W_illum]
        F_rgb/F_ir: [B,C,H,W]
        W_illum: [B,1,h,w], generated by LP. It is resized to [H,W].

    Output:
        F_out: [B,C,H,W]

    The module first decomposes RGB/IR features into shared, RGB-private, and
    IR-private components using FCFS. The shared feature is treated as the
    stable cross-modal representation, while illumination-guided gating chooses
    how much RGB-private or IR-private information should be injected.
    """

    def __init__(
        self,
        c1,
        eps=1e-6,
        tau=0.25,
        temp=0.2,
        share_scale=0.7,
        private_boost=1.5,
        gamma_init=0.1,
    ):
        super().__init__(
            eps=eps,
            tau=tau,
            temp=temp,
            share_scale=share_scale,
            private_boost=private_boost,
        )
        self.private_gate = nn.Sequential(
            nn.Conv2d(3 * c1 + 1, c1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0),
        )
        self.private_refine = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2 * c1, c1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.gamma = nn.Parameter(torch.ones(1) * float(gamma_init))

    @staticmethod
    def _align_illumination(W_illum, target):
        if W_illum.ndim != 4 or W_illum.shape[1] != 1:
            raise ValueError(f"ISPF expects W_illum with shape [B,1,H,W], got {W_illum.shape}.")
        if W_illum.shape[-2:] != target.shape[-2:]:
            W_illum = F.interpolate(W_illum, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return W_illum.to(dtype=target.dtype, device=target.device)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError(
                "ISPF expects [F_rgb, F_ir, W_illum], "
                f"but got {type(x)} with length {len(x) if isinstance(x, (list, tuple)) else 'N/A'}."
            )

        F_rgb, F_ir, W_illum = x
        F_shared, F_rgb_private, F_ir_private = self.freq_coherence_split(F_rgb, F_ir)
        W_illum = self._align_illumination(W_illum, F_shared)

        gate_input = torch.cat([F_shared, F_rgb_private, F_ir_private, W_illum], dim=1)
        gate = torch.sigmoid(self.private_gate(gate_input))
        F_private = gate * F_rgb_private + (1.0 - gate) * F_ir_private
        F_private = self.private_refine(F_private)

        F_fused = self.fusion_conv(torch.cat([F_shared, F_private], dim=1))
        return F_fused + self.gamma * F_private


class LP(nn.Module):
    """
    Light Prior encoder.

    LP estimates an RGB illumination reliability map from the visible RGB
    channels. For a 640x640 input, the output resolution is 80x80, matching the
    P3 feature scale used by the illumination-guided fusion modules.

    Input:
        RGB or RGBT image tensor with shape [B,3/4,H,W]. Only the first three
        visible RGB channels are used.

    Output:
        W_illum: illumination reliability map with shape [B,1,H/8,W/8] and
        values in [0,1]. Larger values indicate stronger RGB reliability.
    """

    def __init__(self):
        super().__init__()

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Project the encoded illumination feature to a single reliability map.
        self.conv4 = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    @staticmethod
    def rgb_to_luminance(x):
        """
        Convert RGB/RGBT input to a BT.709 luminance map.

        Args:
            x: RGB or RGBT image tensor with shape [B,3/4,H,W].

        Returns:
            Luminance tensor with shape [B,1,H,W].
        """
        if x.ndim != 4 or x.shape[1] not in (3, 4):
            raise ValueError(
                f"LP expects RGB/RGBT input with shape [B,3/4,H,W], got {x.shape}."
            )

        x = x[:, :3, :, :]
        r = x[:, 0:1, :, :]
        g = x[:, 1:2, :, :]
        b = x[:, 2:3, :, :]

        L = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return L

    def forward(self, x):
        """
        Args:
            x: RGB or RGBT image tensor with shape [B,3/4,H,W].

        Returns:
            W_illum: illumination reliability map with shape [B,1,H/8,W/8].
        """
        L = self.rgb_to_luminance(x)

        # H,W -> H/2,W/2
        x = self.pool(L)
        x = self.conv1(x)

        # H/2,W/2 -> H/4,W/4
        x = self.pool(x)
        x = self.conv2(x)

        # H/4,W/4 -> H/8,W/8
        x = self.pool(x)
        x = self.conv3(x)

        W_illum = torch.sigmoid(self.conv4(x))
        return W_illum


class ResidualIlluminationFusion(nn.Module):
    """
    Residual illumination-guided RGB-IR fusion.

    This is the original illumination-prior fusion module. It uses the LP output
    as a soft reliability weight: RGB features are emphasized by W_illum, while
    IR features are emphasized by 1 - W_illum. The illumination-guided branch is
    then added as a learnable residual to a baseline RGB/IR fusion branch.

    Input:
        [F_vis, F_ir, W_illum]
        F_vis: visible feature map with shape [B,C,H,W].
        F_ir: infrared feature map with shape [B,C,H,W].
        W_illum: illumination reliability map with shape [B,1,h,w].

    Output:
        F_out: fused feature map with shape [B,C,H,W].
    """

    def __init__(self, in_channels):
        super().__init__()
        self.fuse_conv1 = nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=1, padding=0)
        self.fuse_act = nn.SiLU(inplace=True)
        self.fuse_conv3 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, padding=1)

        self.base_conv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Args:
            x: [F_vis, F_ir, W_illum].

        Returns:
            F_out: fused feature map with shape [B,C,H,W].
        """
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
        F_ir_g = (1.0 - W_illum) * F_ir

        F_cat = torch.cat([F_vis_g, F_ir_g], dim=1)
        F_illu = self.fuse_conv1(F_cat)
        F_illu = self.fuse_act(F_illu)
        F_illu = self.fuse_conv3(F_illu)

        F_base = self.base_conv(torch.cat([F_vis, F_ir], dim=1))
        F_out = F_base + self.gamma * F_illu
        return F_out

###LIFAdd
class LIF(nn.Module):
    # forward鏉╂柨娲栨稉鈧稉?0*80閻ㄥ嫭娼堥柌宥呮禈
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
