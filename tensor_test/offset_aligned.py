from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import DeformConv2d


class SpatialAttention(nn.Module):
    """
    Spatial attention:
        F^p = sigmoid(Conv7x7([Max_c(F), Mean_c(F)])) * F

    Input:
        x: [B, C, H, W]
    Output:
        enhanced: [B, C, H, W]
        spatial_map: [B, 1, H, W]
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()

        if kernel_size not in (3, 7):
            raise ValueError("kernel_size should normally be 3 or 7.")

        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)

        descriptor = torch.cat([max_map, avg_map], dim=1)
        spatial_map = torch.sigmoid(self.conv(descriptor))

        enhanced = x * spatial_map
        return enhanced, spatial_map


class DifferenceChannelEnhancement(nn.Module):
    """
    Channel enhancement based on the concatenated RGB-IR difference feature.

    Important:
    F_dif has 2C channels. Therefore, the generated channel weights also
    contain 2C channels and are split into:
        M_rgb: [B, C, 1, 1]
        M_ir:  [B, C, 1, 1]

    This is more dimensionally rigorous than directly multiplying one 2C
    channel weight map with two C-channel modal features.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive.")

        diff_channels = 2 * channels
        hidden_channels = max(diff_channels // reduction, 8)

        # Shared MLP implemented by 1x1 convolutions.
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(
                diff_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                diff_channels,
                kernel_size=1,
                bias=False,
            ),
        )

        self.channels = channels

    def forward(
        self,
        f_rgb_p: Tensor,
        f_ir_p: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Concatenation-based cross-modal difference representation.
        f_dif = torch.cat([f_rgb_p, f_ir_p], dim=1)

        avg_descriptor = torch.mean(
            f_dif,
            dim=(2, 3),
            keepdim=True,
        )
        max_descriptor = torch.amax(
            f_dif,
            dim=(2, 3),
            keepdim=True,
        )

        channel_map = torch.sigmoid(
            self.shared_mlp(avg_descriptor)
            + self.shared_mlp(max_descriptor)
        )

        # Split the 2C-dimensional channel weights into two modal weights.
        m_rgb, m_ir = torch.split(
            channel_map,
            [self.channels, self.channels],
            dim=1,
        )

        # Residual channel enhancement.
        f_rgb_e = f_rgb_p * m_rgb + f_rgb_p
        f_ir_e = f_ir_p * m_ir + f_ir_p

        return f_rgb_e, f_ir_e, f_dif, m_rgb, m_ir


class OffsetPriorPredictor(nn.Module):
    """
    Predict the explicit initial offset prior Δp^0.

    Input:
        concat(F_rgb^e, F_ir^e): [B, 2C, H, W]

    Output:
        offset_prior: [B, 2K, H, W]

    For a 3x3 deformable kernel:
        K = 9
        2K = 18
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        hidden_ratio: float = 0.5,
        max_prior_offset: float | None = None,
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.num_points = kernel_size * kernel_size
        self.max_prior_offset = max_prior_offset

        hidden_channels = max(int(2 * channels * hidden_ratio), 16)

        self.body = nn.Sequential(
            nn.Conv2d(
                2 * channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )

        self.offset_head = nn.Conv2d(
            hidden_channels,
            2 * self.num_points,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # Start from regular sampling: Δp^0 = 0.
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

    def forward(self, f_rgb_e: Tensor, f_ir_e: Tensor) -> Tensor:
        fused = torch.cat([f_rgb_e, f_ir_e], dim=1)
        offset_prior = self.offset_head(self.body(fused))

        # Optional constraint for training stability.
        if self.max_prior_offset is not None:
            offset_prior = (
                torch.tanh(offset_prior) * self.max_prior_offset
            )

        return offset_prior


class OffsetRefinementAndMask(nn.Module):
    """
    Predict:
        1. residual offset Δp: [B, 2K, H, W]
        2. modulation mask Δm: [B, K, H, W]

    Total output channels:
        2K + K = 3K

    For K=9:
        residual offset channels = 18
        mask channels = 9
        total = 27
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        max_residual_offset: float | None = None,
    ) -> None:
        super().__init__()

        self.num_points = kernel_size * kernel_size
        self.max_residual_offset = max_residual_offset

        self.predictor = nn.Conv2d(
            in_channels=2 * channels,
            out_channels=3 * self.num_points,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # Initial residual offset = 0.
        # Initial raw mask = 0 -> sigmoid(0) = 0.5.
        nn.init.zeros_(self.predictor.weight)
        nn.init.zeros_(self.predictor.bias)

    def forward(
        self,
        f_rgb_e: Tensor,
        f_ir_e: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        fused = torch.cat([f_rgb_e, f_ir_e], dim=1)
        prediction = self.predictor(fused)

        residual_offset, mask_logits = torch.split(
            prediction,
            [2 * self.num_points, self.num_points],
            dim=1,
        )

        if self.max_residual_offset is not None:
            residual_offset = (
                torch.tanh(residual_offset)
                * self.max_residual_offset
            )

        modulation_mask = torch.sigmoid(mask_logits)

        return residual_offset, modulation_mask, mask_logits


class OffsetGuidedDeformableAlignment(nn.Module):
    """
    Full IR-to-RGB alignment module.

    Since the GT boxes are annotated in the RGB coordinate system, this module
    uses RGB as the reference modality and deformably samples IR features into
    the RGB feature coordinate system.

    Pipeline:
        IR/RGB
          -> Spatial Attention
          -> Concatenation Difference
          -> Channel Enhancement
          -> Explicit Offset Prior Δp^0
          -> Residual Offset Δp and Mask Δm
          -> Deformable Alignment
          -> Aligned IR Feature

    Mathematical form:
        F_rgb^a(p) =
            sum_k w_k *
            F_rgb(p + p_k + Δp_k^0 + Δp_k) *
            Δm_k
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        reduction: int = 16,
        max_prior_offset: float | None = None,
        max_residual_offset: float | None = None,
        use_offset_prior: bool = True,
        use_alignment_residual: bool = False,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        padding = kernel_size // 2

        self.spatial_rgb = SpatialAttention(kernel_size=7)
        self.spatial_ir = SpatialAttention(kernel_size=7)

        self.channel_enhancement = DifferenceChannelEnhancement(
            channels=channels,
            reduction=reduction,
        )

        self.offset_prior_predictor = OffsetPriorPredictor(
            channels=channels,
            kernel_size=kernel_size,
            max_prior_offset=max_prior_offset,
        )

        self.refinement_and_mask = OffsetRefinementAndMask(
            channels=channels,
            kernel_size=kernel_size,
            max_residual_offset=max_residual_offset,
        )

        self.deformable_conv = DeformConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.SiLU(inplace=True)

        self.use_alignment_residual = use_alignment_residual
        self.use_offset_prior = use_offset_prior

    @staticmethod
    def _validate_inputs(f_ir: Tensor, f_rgb: Tensor) -> None:
        if f_ir.ndim != 4 or f_rgb.ndim != 4:
            raise ValueError(
                "Inputs must be BCHW tensors."
            )

        if f_ir.shape != f_rgb.shape:
            raise ValueError(
                "IR and RGB features must have identical shapes. "
                f"Got IR={tuple(f_ir.shape)}, "
                f"RGB={tuple(f_rgb.shape)}."
            )

    def forward(
        self,
        f_ir: Tensor,
        f_rgb: Tensor,
        return_auxiliary: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        self._validate_inputs(f_ir, f_rgb)

        # ------------------------------------------------------------
        # Step 1: spatial enhancement
        # ------------------------------------------------------------
        f_ir_p, spatial_ir = self.spatial_ir(f_ir)
        f_rgb_p, spatial_rgb = self.spatial_rgb(f_rgb)

        # ------------------------------------------------------------
        # Step 2-3: concat difference + channel enhancement
        # ------------------------------------------------------------
        (
            f_rgb_e,
            f_ir_e,
            f_dif,
            channel_rgb,
            channel_ir,
        ) = self.channel_enhancement(f_rgb_p, f_ir_p)

        # ------------------------------------------------------------
        # Step 4: explicit offset prior
        # Δp^0: [B, 2K, H, W]
        # ------------------------------------------------------------
        offset_prior = self.offset_prior_predictor(
            f_rgb_e,
            f_ir_e,
        )

        # ------------------------------------------------------------
        # Predict residual offset and modulation mask
        # Δp: [B, 2K, H, W]
        # Δm: [B, K, H, W]
        # ------------------------------------------------------------
        (
            residual_offset,
            modulation_mask,
            mask_logits,
        ) = self.refinement_and_mask(f_rgb_e, f_ir_e)

        # Final sampling offset:
        # Δp_total = Δp^0 + Δp
        total_offset = offset_prior + residual_offset if self.use_offset_prior else residual_offset

        # ------------------------------------------------------------
        # Step 5: IR -> RGB deformable alignment
        #
        # The deformable operator samples only from f_ir.
        # f_rgb guides offset prediction as the RGB-coordinate reference.
        # ------------------------------------------------------------
        f_ir_aligned = self.deformable_conv(
            f_ir,
            total_offset,
            modulation_mask,
        )

        f_ir_aligned = self.activation(
            self.norm(f_ir_aligned)
        )

        # Optional residual connection.
        # Use cautiously because raw IR may still be spatially misaligned.
        if self.use_alignment_residual:
            f_ir_aligned = f_ir_aligned + f_ir

        if not return_auxiliary:
            return f_ir_aligned

        auxiliary = {
            "f_rgb_spatial": f_rgb_p,
            "f_ir_spatial": f_ir_p,
            "spatial_map_rgb": spatial_rgb,
            "spatial_map_ir": spatial_ir,
            "difference_feature": f_dif,
            "f_rgb_enhanced": f_rgb_e,
            "f_ir_enhanced": f_ir_e,
            "channel_weight_rgb": channel_rgb,
            "channel_weight_ir": channel_ir,
            "offset_prior": offset_prior,
            "residual_offset": residual_offset,
            "total_offset": total_offset,
            "modulation_mask": modulation_mask,
            "mask_logits": mask_logits,
        }

        return f_ir_aligned, auxiliary

def test_alignment_module() -> None:
    batch_size = 2
    channels = 128
    height = 80
    width = 80

    f_rgb = torch.randn(
        batch_size,
        channels,
        height,
        width,
    )

    f_ir = torch.randn(
        batch_size,
        channels,
        height,
        width,
    )

    module = OffsetGuidedDeformableAlignment(
        channels=channels,
        kernel_size=3,
        reduction=16,
        # 可根据特征层尺度限制最大偏移。
        max_prior_offset=4.0,
        max_residual_offset=2.0,
        use_offset_prior=True,
        use_alignment_residual=False,
    )

    f_ir_aligned, aux = module(
        f_ir,
        f_rgb,
        return_auxiliary=True,
    )

    print("IR source:       ", f_ir.shape)
    print("RGB reference:   ", f_rgb.shape)
    print("Aligned IR:      ", f_ir_aligned.shape)
    print("Offset prior:    ", aux["offset_prior"].shape)
    print("Residual offset: ", aux["residual_offset"].shape)
    print("Total offset:    ", aux["total_offset"].shape)
    print("Modulation mask: ", aux["modulation_mask"].shape)

    assert f_ir_aligned.shape == f_ir.shape
    assert aux["offset_prior"].shape == (
        batch_size,
        18,
        height,
        width,
    )
    assert aux["residual_offset"].shape == (
        batch_size,
        18,
        height,
        width,
    )
    assert aux["modulation_mask"].shape == (
        batch_size,
        9,
        height,
        width,
    )

    module_without_prior = OffsetGuidedDeformableAlignment(
        channels=channels,
        kernel_size=3,
        reduction=16,
        max_prior_offset=4.0,
        max_residual_offset=2.0,
        use_offset_prior=False,
        use_alignment_residual=False,
    )
    f_ir_aligned_without_prior, aux_without_prior = module_without_prior(
        f_ir,
        f_rgb,
        return_auxiliary=True,
    )
    print("Aligned IR w/o prior:", f_ir_aligned_without_prior.shape)
    assert f_ir_aligned_without_prior.shape == f_ir.shape
    assert torch.equal(
        aux_without_prior["total_offset"],
        aux_without_prior["residual_offset"],
    )


if __name__ == "__main__":
    test_alignment_module()
