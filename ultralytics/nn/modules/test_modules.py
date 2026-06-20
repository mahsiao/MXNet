import torch
from einops import rearrange
from torch import nn


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
