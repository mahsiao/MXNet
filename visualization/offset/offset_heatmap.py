"""Visualize OffsetGuidedDeformableAlignment offsets and alignment quality.

The paired IR path is inferred by replacing ``visible`` with ``infrared``.
Besides a PNG summary, the script saves raw arrays as compressed NPZ files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import DualYOLO
from ultralytics.nn.modules import OffsetGuidedDeformableAlignment
from visualization.feature_heatmap import collect_sources, get_pytorch_model, load_rgbt_tensor


def find_alignment_modules(torch_model):
    modules = [(n, m) for n, m in torch_model.named_modules() if isinstance(m, OffsetGuidedDeformableAlignment)]
    if not modules:
        raise RuntimeError("No OffsetGuidedDeformableAlignment found; use an offset-alignment checkpoint.")
    return modules


def register_capture(name, module, records):
    """Capture module I/O, offset prior, residual offset, and modulation mask."""
    state = {}

    def prior_hook(_, __, output):
        state["prior"] = output.detach().float().cpu()

    def refine_hook(_, __, output):
        residual, mask, _ = output
        state["residual"] = residual.detach().float().cpu()
        state["mask"] = mask.detach().float().cpu()

    def module_hook(alignment, inputs, output):
        f_ir, f_rgb = inputs[0]
        prior = state.pop("prior")
        residual = state.pop("residual")
        records[name] = {
            "ir": f_ir.detach().float().cpu(),
            "rgb": f_rgb.detach().float().cpu(),
            "aligned": output.detach().float().cpu(),
            "prior": prior,
            "residual": residual,
            "total": prior + residual if alignment.use_offset_prior else residual,
            "mask": state.pop("mask"),
        }

    return [
        module.offset_prior_predictor.register_forward_hook(prior_hook),
        module.refinement_and_mask.register_forward_hook(refine_hook),
        module.register_forward_hook(module_hook),
    ]


def offset_components(offset):
    """Return mean dx/dy and mean/max point magnitudes from [2K,H,W]."""
    if offset.ndim != 3 or offset.shape[0] % 2:
        raise ValueError(f"Expected [2K,H,W] offset, got {tuple(offset.shape)}")
    pairs = offset.reshape(-1, 2, *offset.shape[-2:])
    dy, dx = pairs[:, 0], pairs[:, 1]  # torchvision offset order is (dy, dx)
    point_magnitude = torch.sqrt(dx.square() + dy.square())
    return dx.mean(0).numpy(), dy.mean(0).numpy(), point_magnitude.mean(0).numpy(), point_magnitude.max(0).values.numpy()


def cosine_map(a, b):
    return F.cosine_similarity(a, b, dim=0, eps=1e-6).numpy()


def robust_normalize(array, percentile):
    array = np.nan_to_num(array)
    upper = float(np.percentile(array, percentile))
    if upper <= 1e-12:
        return np.zeros_like(array, dtype=np.float32), upper
    return np.clip(array / upper, 0, 1).astype(np.float32), upper


def display_images(image_tensor):
    array = image_tensor[0].detach().float().cpu().clamp(0, 1).numpy()
    return np.transpose(array[:3], (1, 2, 0)), array[3]


def resolve_device(device):
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    value = str(device).strip()
    if value.isdigit():
        value = f"cuda:{value}"
    return torch.device(value)


def overlay(image, heatmap, alpha):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    color = plt.get_cmap("turbo")(heatmap)[..., :3]
    return np.clip((1 - alpha) * image + alpha * color, 0, 1)


def draw_quiver(ax, image, dx, dy, step):
    h, w = dx.shape
    ys, xs = np.arange(0, h, max(step, 1)), np.arange(0, w, max(step, 1))
    grid_x, grid_y = np.meshgrid(xs, ys)
    ax.imshow(image, extent=(0, w, h, 0))
    ax.quiver(grid_x, grid_y, dx[np.ix_(ys, xs)], dy[np.ix_(ys, xs)], color="cyan",
              angles="xy", scale_units="xy", scale=1, width=0.004, headwidth=3.5)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)


def save_result(record, image_tensor, stem, module_name, alpha, percentile, quiver_step):
    rgb_image, ir_image = display_images(image_tensor)
    total, prior, residual = record["total"][0], record["prior"][0], record["residual"][0]
    mask = record["mask"][0].mean(0).numpy()
    dx, dy, magnitude, max_magnitude = offset_components(total)
    _, _, prior_magnitude, _ = offset_components(prior)
    _, _, residual_magnitude, _ = offset_components(residual)

    before = cosine_map(record["rgb"][0], record["ir"][0])
    after = cosine_map(record["rgb"][0], record["aligned"][0])
    gain = after - before
    magnitude_norm, upper = robust_normalize(magnitude, percentile)
    ir_rgb = np.repeat(ir_image[..., None], 3, axis=2)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    axes[0, 0].imshow(rgb_image); axes[0, 0].set_title("RGB input (reference)")
    axes[0, 1].imshow(ir_image, cmap="gray"); axes[0, 1].set_title("IR input (to align)")
    axes[0, 2].imshow(overlay(rgb_image, magnitude_norm, alpha)); axes[0, 2].set_title(f"Offset magnitude on RGB\nP{percentile:g}={upper:.3f} feature px")
    axes[0, 3].imshow(overlay(ir_rgb, magnitude_norm, alpha)); axes[0, 3].set_title("Offset magnitude on IR")
    draw_quiver(axes[1, 0], rgb_image, dx, dy, quiver_step); axes[1, 0].set_title("Mean sampling displacement (dx, dy)")
    plot = axes[1, 1].imshow(mask, cmap="viridis", vmin=0, vmax=1)
    axes[1, 1].set_title(f"Modulation mask (mean={mask.mean():.3f})"); fig.colorbar(plot, ax=axes[1, 1], fraction=0.046)
    plot = axes[1, 2].imshow(before, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 2].set_title(f"RGB-IR similarity before={before.mean():.3f}"); fig.colorbar(plot, ax=axes[1, 2], fraction=0.046)
    limit = max(float(np.percentile(np.abs(gain), 99)), 1e-3)
    plot = axes[1, 3].imshow(gain, cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[1, 3].set_title(f"Similarity gain (after-before)\nafter={after.mean():.3f}, gain={gain.mean():+.3f}")
    fig.colorbar(plot, ax=axes[1, 3], fraction=0.046)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"OffsetGuidedDeformableAlignment: {module_name}", fontsize=16)
    figure_path = stem.with_suffix(".png")
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(stem.with_suffix(".npz"), dx=dx, dy=dy, offset_magnitude=magnitude,
                        max_point_magnitude=max_magnitude, prior_magnitude=prior_magnitude,
                        residual_magnitude=residual_magnitude, modulation_mask=mask,
                        similarity_before=before, similarity_after=after, similarity_gain=gain)
    return figure_path, magnitude.mean(), max_magnitude.max(), before.mean(), after.mean()


def visualize_offsets(model, source, out_dir="visualization/runs/offset", imgsz=640, device=None,
                      alpha=0.45, percentile=99.0, quiver_step=5):
    sources = collect_sources(source)
    if not sources:
        raise FileNotFoundError(f"No visible image found from: {source}")
    yolo = model if isinstance(model, DualYOLO) else DualYOLO(str(model))
    torch_model = get_pytorch_model(yolo)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100].")
    device = resolve_device(device)
    torch_model.to(device).eval()
    modules, records, handles = find_alignment_modules(torch_model), {}, []
    for name, module in modules:
        handles.extend(register_capture(name, module, records))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    try:
        with torch.inference_mode():
            for source_path in sources:
                records.clear()
                image_tensor = load_rgbt_tensor(source_path, imgsz=imgsz, device=device)
                torch_model(image_tensor)
                for index, (name, _) in enumerate(modules):
                    if name not in records:
                        continue
                    safe_name = name.replace(".", "_") or f"alignment_{index}"
                    result = save_result(records[name], image_tensor, out_dir / f"{source_path.stem}_{safe_name}",
                                         name, alpha, percentile, quiver_step)
                    outputs.append(result[0])
                    print(f"Saved {result[0]} | offset mean/max={result[1]:.3f}/{result[2]:.3f} feature px | similarity {result[3]:.3f}->{result[4]:.3f}")
    finally:
        for handle in handles:
            handle.remove()
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize deformable alignment offsets and RGB-IR consistency.")
    parser.add_argument("--model", required=True, help="Trained offset-alignment checkpoint (.pt).")
    parser.add_argument("--source", required=True, help="Visible image, directory, or glob.")
    parser.add_argument("--out-dir", default="visualization/runs/offset")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="For example: 0, cuda:0, or cpu.")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--quiver-step", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_offsets(args.model, args.source, args.out_dir, args.imgsz, args.device,
                      args.alpha, args.percentile, args.quiver_step)
