import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules.fusion import FCFS, FCFSPrivateResidual, FCFSShared  # noqa: E402


VISIBLE_PATH = Path(r"E:\CProject\Dataset\DVOBB\train\visible\images\00090.jpg")
INFRARED_PATH = Path(r"E:\CProject\Dataset\DVOBB\train\infrared\images\00090.jpg")
CKPT_PATH = Path(r"D:\CProject\MXNet\eval\ckpts\DV\DV-11n-mid-fcfs-private-residual.pt")
OUTPUT_DIR = ROOT / "visualization" / "fcfs90" / "00090" / "heatmap"
OUTPUT_PATH = OUTPUT_DIR / "fcfs_00090_fcfs_mean_p3_spatial_heatmaps.png"
OVERLAY_OUTPUT_PATH = OUTPUT_DIR / "fcfs_00090_fcfs_mean_p3_spatial_heatmap_overlays.png"

HEATMAP_TITLES = ["Shared", "RGB-Private", "IR-Private"]


def load_rgb_ir_tensor(size=640, device="cpu"):
    """Load paired RGB/IR images as one 4-channel YOLO input tensor."""
    rgb = Image.open(VISIBLE_PATH).convert("RGB").resize((size, size), Image.BILINEAR)
    ir = Image.open(INFRARED_PATH).convert("L").resize((size, size), Image.BILINEAR)

    rgb_arr = np.asarray(rgb).astype(np.float32) / 255.0
    ir_arr = np.asarray(ir).astype(np.float32) / 255.0

    rgb_tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1)
    ir_tensor = torch.from_numpy(ir_arr)[None]
    return torch.cat([rgb_tensor, ir_tensor], dim=0)[None].to(device)


def load_display_images(size=640):
    """Load RGB and IR images for visualization panels."""
    rgb = Image.open(VISIBLE_PATH).convert("RGB").resize((size, size), Image.BILINEAR)
    ir = Image.open(INFRARED_PATH).convert("L").resize((size, size), Image.BILINEAR).convert("RGB")
    return np.asarray(rgb), np.asarray(ir)


def find_fcfs_layers(model):
    """Find FCFS-style fusion layers in the YOLO model."""
    target_types = (FCFS, FCFSShared, FCFSPrivateResidual)
    return [(name, module) for name, module in model.named_modules() if isinstance(module, target_types)]


def capture_p3_inputs(model, image_tensor):
    """Capture RGB/IR feature inputs of the first FCFS layer, which corresponds to P3."""
    layers = find_fcfs_layers(model)
    if not layers:
        raise RuntimeError("No FCFS/FCFSShared/FCFSPrivateResidual layer found in the loaded model.")

    layer_name, layer = layers[0]
    captured = {}

    def hook_fn(module, inputs, output):
        layer_input = inputs[0]
        if not isinstance(layer_input, (list, tuple)) or len(layer_input) != 2:
            raise RuntimeError(f"Expected FCFS input [F_rgb, F_ir], got {type(layer_input)}.")
        captured["rgb"] = layer_input[0].detach()
        captured["ir"] = layer_input[1].detach()

    handle = layer.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        _ = model(image_tensor)
    handle.remove()

    if "rgb" not in captured or "ir" not in captured:
        raise RuntimeError(f"Failed to capture P3 features from {layer_name}.")
    return layer_name, layer, captured["rgb"], captured["ir"]


def fcfs_spatial_components(x, y, layer):
    """Compute Shared/RGB-private/IR-private spatial features with FCFS."""
    eps = layer.eps
    spatial_size = x.shape[-2:]
    dtype = x.dtype

    with torch.cuda.amp.autocast(enabled=False):
        x = x.float()
        y = y.float()

        X = torch.fft.rfft2(x, norm="ortho")
        Y = torch.fft.rfft2(y, norm="ortho")

        Sxx = (X.real ** 2 + X.imag ** 2) + eps
        Syy = (Y.real ** 2 + Y.imag ** 2) + eps
        Sxy = X * torch.conj(Y)

        coh = (Sxy.real ** 2 + Sxy.imag ** 2) / (Sxx * Syy)
        balance = torch.sqrt(Sxx * Syy) / (Sxx + Syy + eps)

        Mc = torch.sigmoid(((coh * balance) - layer.tau) / layer.temp)
        Mc = Mc * layer.share_scale + (1.0 - layer.share_scale) * 0.5
        Mc = Mc.clamp(0.0, 1.0)

        diff_mag = torch.abs(Sxx - Syy) / (Sxx + Syy + eps)
        Mr = (1.0 - Mc) * diff_mag * (Sxx / (Sxx + Syy))
        Mi = (1.0 - Mc) * diff_mag * (Syy / (Sxx + Syy))

        Mr = Mr * layer.private_boost
        Mi = Mi * layer.private_boost

        denom = Mc + Mr + Mi + eps
        Mc = Mc / denom
        Mr = Mr / denom
        Mi = Mi / denom

        # Keep this consistent with the frequency-copy visualization script.
        Zc_hat = Mc * (X + Y)
        Zr_hat = Mr * X
        Zi_hat = Mi * Y

        Zc = torch.fft.irfft2(Zc_hat, s=spatial_size, norm="ortho")
        Zr = torch.fft.irfft2(Zr_hat, s=spatial_size, norm="ortho")
        Zi = torch.fft.irfft2(Zi_hat, s=spatial_size, norm="ortho")

    return [Zc.to(dtype).detach(), Zr.to(dtype).detach(), Zi.to(dtype).detach()]


def feature_to_heatmap(feat):
    """Convert [B,C,H,W] feature tensor to one normalized spatial heatmap."""
    feat = feat.detach().float().cpu()
    heatmap = feat.abs().mean(dim=(0, 1)).numpy()
    lo, hi = np.percentile(heatmap, [1, 99])
    return np.clip((heatmap - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def resize_and_smooth_heatmap(heatmap, size=(640, 640), blur_radius=1.2):
    """Upsample a low-resolution feature heatmap and apply light display smoothing."""
    heat_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    heat_img = heat_img.resize(size, Image.BICUBIC)
    if blur_radius > 0:
        heat_img = heat_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.asarray(heat_img).astype(np.float32) / 255.0


def overlay_heatmap(base_image, heatmap, alpha=0.45):
    """Overlay a normalized heatmap on an RGB image."""
    heat_color = (cm.jet(heatmap)[..., :3] * 255).astype(np.uint8)
    base = base_image.astype(np.float32)
    overlay = (1.0 - alpha) * base + alpha * heat_color.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_heatmaps(heatmaps, layer_name):
    """Save Shared/RGB-private/IR-private spatial heatmaps."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)
    for ax, title, heatmap in zip(axes, HEATMAP_TITLES, heatmaps):
        im = ax.imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    fig.suptitle(f"FCFS spatial heatmaps (P3 {layer_name}, channel=mean-abs)", fontsize=13)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_overlay_heatmaps(heatmaps, rgb_image, ir_image, layer_name):
    """Save original images and FCFS heatmaps overlaid on the relevant modality."""
    target_size = (rgb_image.shape[1], rgb_image.shape[0])
    shared = resize_and_smooth_heatmap(heatmaps[0], size=target_size)
    rgb_private = resize_and_smooth_heatmap(heatmaps[1], size=target_size)
    ir_private = resize_and_smooth_heatmap(heatmaps[2], size=target_size)

    panels = [
        ("RGB", rgb_image),
        ("IR", ir_image),
        ("Shared on RGB", overlay_heatmap(rgb_image, shared)),
        ("RGB-Private on RGB", overlay_heatmap(rgb_image, rgb_private)),
        ("IR-Private on IR", overlay_heatmap(ir_image, ir_private)),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), constrained_layout=True)
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle(f"FCFS spatial heatmap overlays (P3 {layer_name}, channel=mean-abs)", fontsize=13)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OVERLAY_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(CKPT_PATH)).model.to(device)
    image_tensor = load_rgb_ir_tensor(device=device)

    layer_name, layer, rgb_feat, ir_feat = capture_p3_inputs(model, image_tensor)
    components = fcfs_spatial_components(rgb_feat, ir_feat, layer)
    heatmaps = [feature_to_heatmap(comp) for comp in components]
    rgb_image, ir_image = load_display_images()
    save_heatmaps(heatmaps, layer_name)
    save_overlay_heatmaps(heatmaps, rgb_image, ir_image, layer_name)

    print(f"Saved P3 FCFS spatial heatmaps to: {OUTPUT_PATH}")
    print(f"Saved P3 FCFS spatial heatmap overlays to: {OVERLAY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
