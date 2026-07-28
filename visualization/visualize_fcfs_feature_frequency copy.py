import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules.fusion import FCFS, FCFSPrivateResidual, FCFSShared  # noqa: E402


VISIBLE_PATH = Path(r"E:\CProject\Dataset\DVOBB\train\visible\images\00090.jpg")
INFRARED_PATH = Path(r"E:\CProject\Dataset\DVOBB\train\infrared\images\00090.jpg")
CKPT_PATH = Path(r"D:\CProject\MXNet\eval\ckpts\DV\DV-11n-mid-fcfs-private-residual.pt")
OUTPUT_PATH = ROOT / "visualization" / "fcfs90" / "00090" / "mean" / "fcfs_00090_fcfs_mean_p3_spectra_shared_scale.png"

COMPONENT_TITLES = ["RGB", "IR", "Shared", "RGB-Private", "IR-Private"]


def load_rgb_ir_tensor(size=640, device="cpu"):
    """Load paired RGB/IR images as one 4-channel YOLO input tensor."""
    rgb = Image.open(VISIBLE_PATH).convert("RGB").resize((size, size), Image.BILINEAR)
    ir = Image.open(INFRARED_PATH).convert("L").resize((size, size), Image.BILINEAR)

    rgb_arr = np.asarray(rgb).astype(np.float32) / 255.0
    ir_arr = np.asarray(ir).astype(np.float32) / 255.0

    rgb_tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1)
    ir_tensor = torch.from_numpy(ir_arr)[None]
    return torch.cat([rgb_tensor, ir_tensor], dim=0)[None].to(device)


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


def fcfs_frequency_components(x, y, layer):
    """Compute RGB/IR/shared/private frequency components with the FCFS masks."""
    eps = layer.eps
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

        # Keep the current copy-file behavior: shared spectrum uses Mc * (X + Y).
        Zc_hat = Mc * (X + Y)
        Zr_hat = Mr * X
        Zi_hat = Mi * Y

    return [X.detach(), Y.detach(), Zc_hat.detach(), Zr_hat.detach(), Zi_hat.detach()]


def rfft_to_square_spectrum(component, spatial_size):
    """Convert an rfft2 component to a centered full fft2 log-magnitude map."""
    component = component.detach().cpu()
    spatial = torch.fft.irfft2(component, s=spatial_size, norm="ortho")
    full_fft = torch.fft.fft2(spatial, norm="ortho")
    mag = torch.log1p(torch.abs(full_fft))
    mag = torch.fft.fftshift(mag, dim=(-2, -1))
    return mag.mean(dim=(0, 1)).numpy()


def normalize_shared(maps):
    """Normalize five spectra with one shared color range for fair comparison."""
    stacked = np.stack(maps, axis=0)
    lo, hi = np.percentile(stacked, [1, 99])
    return [np.clip((m - lo) / (hi - lo + 1e-8), 0.0, 1.0) for m in maps]


def save_p3_shared_scale_figure(maps, layer_name):
    """Save the five-column P3 FCFS frequency visualization."""
    maps = normalize_shared(maps)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), constrained_layout=True)
    for ax, title, spec in zip(axes, COMPONENT_TITLES, maps):
        im = ax.imshow(spec, cmap="turbo", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.suptitle(f"FCFS frequency spectra, shared Turbo scale (P3 {layer_name}, channel=mean)", fontsize=13)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(CKPT_PATH)).model.to(device)
    image_tensor = load_rgb_ir_tensor(device=device)

    layer_name, layer, rgb_feat, ir_feat = capture_p3_inputs(model, image_tensor)
    components = fcfs_frequency_components(rgb_feat, ir_feat, layer)
    maps = [rfft_to_square_spectrum(comp, rgb_feat.shape[-2:]) for comp in components]
    save_p3_shared_scale_figure(maps, layer_name)

    print(f"Saved P3 FCFS shared-scale spectrum to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
