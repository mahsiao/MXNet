import numpy as np
import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F

from .block import C2f, C3k, C3k2
from .conv import Conv


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                      groups=hidden_features),
            act_layer()
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

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
######################################## IR-AFAB start ########################################

class AdaptiveMultiKernelConv2d(nn.Module):

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        self.dwconv = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, square_kernel_size, padding=square_kernel_size//2, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=in_channels)
        ])
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()
        
        # Dynamic Kernel Weights
        self.dkw = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels * 3, 1)
        )
        
    def forward(self, x):
        x_dkw = rearrange(self.dkw(x), 'bs (g ch) h w -> g bs ch h w', g=3)
        x_dkw = F.softmax(x_dkw, dim=0)
        x = torch.stack([self.dwconv[i](x) * x_dkw[i] for i in range(len(self.dwconv))]).sum(0)
        return self.act(self.bn(x))

class AdaptiveMultiKernelDWConv(nn.Module):
    def __init__(self, channel=256, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = channel // 2
        
        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(AdaptiveMultiKernelConv2d(min_ch, ks, ks * 3 + 2))
        self.conv_1x1 = Conv(channel, channel, k=1)
        
    def forward(self, x):
        _, c, _, _ = x.size()
        x_group = torch.split(x, [c // 2, c // 2], dim=1)
        x_group = torch.cat([self.convs[i](x_group[i]) for i in range(len(self.convs))], dim=1)
        x = self.conv_1x1(x_group)
        return x

class AdaptiveMixerBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mixer = AdaptiveMultiKernelDWConv(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = ConvolutionalGLU(dim)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class AFAB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(AdaptiveMixerBlock(c_) for _ in range(n)))

class IR_AFAB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(AFAB(self.c, self.c, 2, shortcut, g) if c3k else AdaptiveMixerBlock(self.c) for _ in range(n))
        
######################################## IR-AFAB end ########################################


######################################## SFEM start ########################################

class ScharrConv(nn.Module):
    def __init__(self, channel):
        super(ScharrConv, self).__init__()

        scharr_kernel_x = np.array([[3, 0, -3],
                                    [10, 0, -10],
                                    [3, 0, -3]], dtype=np.float32)

        scharr_kernel_y = np.array([[3, 10, 3],
                                    [0, 0, 0],
                                    [-3, -10, -3]], dtype=np.float32)

        scharr_kernel_x = torch.tensor(scharr_kernel_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        scharr_kernel_y = torch.tensor(scharr_kernel_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)

        self.scharr_kernel_x = scharr_kernel_x.expand(channel, 1, 3, 3)  # (channel, 1, 3, 3)
        self.scharr_kernel_y = scharr_kernel_y.expand(channel, 1, 3, 3)  # (channel, 1, 3, 3)

        self.scharr_kernel_x_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.scharr_kernel_y_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)

        self.scharr_kernel_x_conv.weight.data = self.scharr_kernel_x.clone()
        self.scharr_kernel_y_conv.weight.data = self.scharr_kernel_y.clone()

        self.scharr_kernel_x_conv.requires_grad = False
        self.scharr_kernel_y_conv.requires_grad = False

    def forward(self, x):
        grad_x = self.scharr_kernel_x_conv(x)
        grad_y = self.scharr_kernel_y_conv(x)

        edge_magnitude = grad_x * 0.5 + grad_y * 0.5
        # edge_magnitude = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2))

        return edge_magnitude


class FreqSpatial(nn.Module):
    def __init__(self, in_channels):
        super(FreqSpatial, self).__init__()

        self.sed = ScharrConv(in_channels)

        self.spatial_conv1 = Conv(in_channels, in_channels)
        self.spatial_conv2 = Conv(in_channels, in_channels)

        self.fft_conv = Conv(in_channels * 2, in_channels * 2, 3)
        self.fft_conv2 = Conv(in_channels, in_channels, 3)

        self.final_conv = Conv(in_channels, in_channels, 1)

    def forward(self, x):
        batch, c, h, w = x.size()

        spatial_feat = self.sed(x)
        spatial_feat = self.spatial_conv1(spatial_feat)
        spatial_feat = self.spatial_conv2(spatial_feat + x)

        fft_feat = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(fft_feat), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(fft_feat), dim=-1)
        fft_feat = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        fft_feat = rearrange(fft_feat, 'b c h w d -> b (c d) h w').contiguous()

        fft_feat = self.fft_conv(fft_feat)

        fft_feat = rearrange(fft_feat, 'b (c d) h w -> b c h w d', d=2).contiguous()
        fft_feat = torch.view_as_complex(fft_feat)
        fft_feat = torch.fft.irfft2(fft_feat, s=(h, w), norm='ortho')

        fft_feat = self.fft_conv2(fft_feat)

        out = spatial_feat + fft_feat
        return self.final_conv(out)


class SFEM(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(FreqSpatial(self.c) for _ in range(n))


######################################## SFEM end ########################################

class CrossAttention_S(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention_S, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,
                                  bias=bias)

        self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        self.qk_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2,
                                   bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        query_fea = x[0]
        context_fea = x[1]
        b, c, h, w = query_fea.shape

        # Cross-modal attention used in the thesis: Q comes from the current modality,
        # while K and V come from the other modality.
        qk_query = self.qk_dwconv(self.qk(query_fea))
        q, _ = qk_query.chunk(2, dim=1)

        qk_context = self.qk_dwconv(self.qk(context_fea))
        _, k = qk_context.chunk(2, dim=1)
        v = self.v_dwconv(self.v(context_fea))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out



class CrossAttention_M(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention_M, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        rgb_fea = x[0]  # 2024/11/1 added by wwc
        ir_fea = x[1]  # 2024/11/1 added by wwc
        b, c, h, w = rgb_fea.shape

        rgb_qkv = self.qkv_dwconv(self.qkv(rgb_fea))
        rgb_q, rgb_k, rgb_v = rgb_qkv.chunk(3, dim=1)

        ir_qkv = self.qkv_dwconv(self.qkv(ir_fea))
        ir_q, ir_k, ir_v = ir_qkv.chunk(3, dim=1)

        rgb_q = rearrange(rgb_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        rgb_k = rearrange(rgb_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        rgb_v = rearrange(rgb_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        ir_q = rearrange(ir_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ir_k = rearrange(ir_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ir_v = rearrange(ir_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        rgb_q = torch.nn.functional.normalize(rgb_q, dim=-1)
        rgb_k = torch.nn.functional.normalize(rgb_k, dim=-1)

        ir_q = torch.nn.functional.normalize(ir_q, dim=-1)
        ir_k = torch.nn.functional.normalize(ir_k, dim=-1)

        # Bidirectional cross-modal attention.
        # RGB output: Q from RGB, K/V from IR.
        # IR output: Q from IR, K/V from RGB.
        attn_rgb = (rgb_q @ ir_k.transpose(-2, -1)) * self.temperature
        attn_ir = (ir_q @ rgb_k.transpose(-2, -1)) * self.temperature

        attn_rgb = attn_rgb.softmax(dim=-1)
        attn_ir = attn_ir.softmax(dim=-1)

        out_rgb = (attn_rgb @ ir_v)
        out_ir = (attn_ir @ rgb_v)

        out_ir = rearrange(out_ir, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_rgb = rearrange(out_rgb, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_ir = self.project_out(out_ir)
        out_rgb = self.project_out(out_rgb)

        return [out_rgb, out_ir]


class IRGuidedRGBEnhancement(nn.Module):
    """Enhance the RGB branch with IR features using one-way cross attention.

    Inputs:
        x[0]: RGB feature, used as Query.
        x[1]: IR feature, used as Key and Value.

    Output:
        RGB-enhanced feature. The cross-modal output is added back to the RGB
        feature through a residual connection to preserve original RGB details.
    """

    def __init__(self, dim, num_heads=8, bias=False):
        super(IRGuidedRGBEnhancement, self).__init__()
        self.mhca_rgb = CrossAttention_S(dim, num_heads, bias)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]

        rgb_cross = self.mhca_rgb([rgb_fea, ir_fea])
        rgb_enhanced = rgb_fea + rgb_cross
        return rgb_enhanced


class RGBGuidedIREnhancement(nn.Module):
    """Enhance the IR branch with RGB features using one-way cross attention.

    Inputs:
        x[0]: RGB feature, used as Key and Value.
        x[1]: IR feature, used as Query.

    Output:
        IR-enhanced feature. The cross-modal output is added back to the IR
        feature through a residual connection to preserve original IR details.
    """

    def __init__(self, dim, num_heads=8, bias=False):
        super(RGBGuidedIREnhancement, self).__init__()
        self.mhca_ir = CrossAttention_S(dim, num_heads, bias)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]

        ir_cross = self.mhca_ir([ir_fea, rgb_fea])
        ir_enhanced = ir_fea + ir_cross
        return ir_enhanced


class IRValueGuidedRGBEnhancement(nn.Module):
    """Enhance RGB with IR values while attention weights are decided by RGB.

    Inputs:
        x[0]: RGB feature, used as Query and Key.
        x[1]: IR feature, used as Value.

    Output:
        RGB-enhanced feature. Q/K from RGB preserve RGB spatial attention,
        while V from IR injects complementary infrared information.
    """

    def __init__(self, dim, num_heads=8, bias=False):
        super(IRValueGuidedRGBEnhancement, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.qk_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]
        b, c, h, w = rgb_fea.shape

        qk = self.qk_dwconv(self.qk(rgb_fea))
        q, k = qk.chunk(2, dim=1)
        v = self.v_dwconv(self.v(ir_fea))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return rgb_fea + out


class RGBValueGuidedIREnhancement(nn.Module):
    """Enhance IR with RGB values while attention weights are decided by IR.

    Inputs:
        x[0]: RGB feature, used as Value.
        x[1]: IR feature, used as Query and Key.

    Output:
        IR-enhanced feature. Q/K from IR preserve IR spatial attention,
        while V from RGB injects complementary visible-spectrum information.
    """

    def __init__(self, dim, num_heads=8, bias=False):
        super(RGBValueGuidedIREnhancement, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.qk_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]
        b, c, h, w = ir_fea.shape

        qk = self.qk_dwconv(self.qk(ir_fea))
        q, k = qk.chunk(2, dim=1)
        v = self.v_dwconv(self.v(rgb_fea))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return ir_fea + out


class BidirectionalKVGuidedEnhancement(nn.Module):
    """Bidirectional cross-modal enhancement with K/V from the other modality.

    Inputs:
        x[0]: RGB feature.
        x[1]: IR feature.

    Branches:
        RGB enhancement: Q=RGB, K/V=IR.
        IR enhancement: Q=IR, K/V=RGB.

    Output:
        Concat([enhanced RGB, enhanced IR]) with 2C channels. This keeps the
        same output shape as the original P3 Concat baseline.
    """

    def __init__(self, dim, num_heads=8, bias=False):
        super(BidirectionalKVGuidedEnhancement, self).__init__()
        self.rgb_enhance = IRGuidedRGBEnhancement(dim, num_heads, bias)
        self.ir_enhance = RGBGuidedIREnhancement(dim, num_heads, bias)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]

        rgb_enhanced = self.rgb_enhance([rgb_fea, ir_fea])
        ir_enhanced = self.ir_enhance([rgb_fea, ir_fea])
        return torch.cat([rgb_enhanced, ir_enhanced], dim=1)


class BidirectionalValueGuidedEnhancement(nn.Module):
    """Bidirectional cross-modal enhancement with Q/K from the current modality.

    Inputs:
        x[0]: RGB feature.
        x[1]: IR feature.

    Branches:
        RGB enhancement: Q/K=RGB, V=IR.
        IR enhancement: Q/K=IR, V=RGB.

    Output:
        Concat([enhanced RGB, enhanced IR]) with 2C channels. This keeps the
        same output shape as the original P3 Concat baseline.
    """

    def __init__(self, dim, num_heads=8, bias=False):
        super(BidirectionalValueGuidedEnhancement, self).__init__()
        self.rgb_enhance = IRValueGuidedRGBEnhancement(dim, num_heads, bias)
        self.ir_enhance = RGBValueGuidedIREnhancement(dim, num_heads, bias)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]

        rgb_enhanced = self.rgb_enhance([rgb_fea, ir_fea])
        ir_enhanced = self.ir_enhance([rgb_fea, ir_fea])
        return torch.cat([rgb_enhanced, ir_enhanced], dim=1)
