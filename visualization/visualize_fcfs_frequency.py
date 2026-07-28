import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_luminance(path, size=None):
    """Load RGB/IR image as one-channel luminance tensor [1,1,H,W]."""
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize(size, Image.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return torch.from_numpy(lum)[None, None]


def fcfs_split(x, y, eps=1e-6, tau=0.25, temp=0.2, share_scale=0.7, private_boost=1.5):
    """
    FCFS from SPFD, adapted for image/feature visualization.

    Args:
        x: RGB luminance or feature map, [B,C,H,W].
        y: IR luminance or feature map, [B,C,H,W].

    Returns:
        shared, rgb_private, ir_private, rgb_fft, ir_fft
    """
    if x.shape != y.shape:
        raise ValueError(f"FCFS expects the same shape, got {x.shape} and {y.shape}.")

    with torch.cuda.amp.autocast(enabled=False):
        x = x.float()
        y = y.float()

        X = torch.fft.rfft2(x, norm="ortho")
        Y = torch.fft.rfft2(y, norm="ortho")

        Sxx = (X.real**2 + X.imag**2) + eps
        Syy = (Y.real**2 + Y.imag**2) + eps
        Sxy = X * torch.conj(Y)

        coherence = (Sxy.real**2 + Sxy.imag**2) / (Sxx * Syy)
        strength_balance = torch.sqrt(Sxx * Syy) / (Sxx + Syy + eps)

        Mc = torch.sigmoid(((coherence * strength_balance) - tau) / temp)
        Mc = Mc * share_scale + (1.0 - share_scale) * 0.5
        Mc = Mc.clamp(0.0, 1.0)

        diff_mag = torch.abs(Sxx - Syy) / (Sxx + Syy + eps)
        Mr = (1.0 - Mc) * diff_mag * (Sxx / (Sxx + Syy))
        Mi = (1.0 - Mc) * diff_mag * (Syy / (Sxx + Syy))

        Mr = Mr * private_boost
        Mi = Mi * private_boost

        denom = Mc + Mr + Mi + eps
        Mc = Mc / denom
        Mr = Mr / denom
        Mi = Mi / denom

        shared_hat = Mc * (X + Y) * 0.5
        rgb_private_hat = Mr * X
        ir_private_hat = Mi * Y

        shared = torch.fft.irfft2(shared_hat, s=x.shape[-2:], norm="ortho")
        rgb_private = torch.fft.irfft2(rgb_private_hat, s=x.shape[-2:], norm="ortho")
        ir_private = torch.fft.irfft2(ir_private_hat, s=x.shape[-2:], norm="ortho")

    return shared, rgb_private, ir_private, X, Y


def spectrum_map(tensor):
    """
    Convert image/feature tensor [B,C,H,W] to a log-magnitude spectrum map.
    The low-frequency component is shifted to the center for visualization.
    """
    fft = torch.fft.fft2(tensor.float(), norm="ortho")
    mag = torch.log1p(torch.abs(fft))
    mag = torch.fft.fftshift(mag, dim=(-2, -1))
    mag = mag.mean(dim=(0, 1)).detach().cpu().numpy()
    return mag


def normalize_shared(maps):
    """Normalize all spectrum maps with one shared range for fair comparison."""
    stacked = np.stack(maps, axis=0)
    lo, hi = np.percentile(stacked, [1, 99])
    return [np.clip((m - lo) / (hi - lo + 1e-8), 0.0, 1.0) for m in maps]


def plot_five_spectra(rgb, ir, shared, rgb_private, ir_private, output):
    names = ["RGB", "IR", "Shared", "RGB-Private", "IR-Private"]
    maps = [
        spectrum_map(rgb),
        spectrum_map(ir),
        spectrum_map(shared),
        spectrum_map(rgb_private),
        spectrum_map(ir_private),
    ]
    maps = normalize_shared(maps)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)
    for ax, name, spec in zip(axes, names, maps):
        ax.imshow(spec, cmap="turbo")
        ax.set_title(name, fontsize=13)
        ax.axis("off")

    fig.suptitle("Frequency-domain visualizations of shared and private features", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize FCFS frequency-domain shared/private separation.")
    parser.add_argument("--rgb", default=r"E:\CProject\Dataset\DVOBB\train\visible\images\00007.jpg", help="Path to the RGB image.")
    parser.add_argument("--ir", default=r"E:\CProject\Dataset\DVOBB\train\infrared\images\00007.jpg", help="Path to the IR image.")
    parser.add_argument("--out", default=r"D:\CProject\MXNet\visualization\fcfs_frequency_00007.png", help="Output figure path.")
    parser.add_argument("--size", type=int, default=512, help="Resize both images to size x size before FCFS.")
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--share-scale", type=float, default=0.7)
    parser.add_argument("--private-boost", type=float, default=1.5)
    args = parser.parse_args()

    size = (args.size, args.size) if args.size > 0 else None
    rgb = load_luminance(args.rgb, size=size)
    ir = load_luminance(args.ir, size=size)

    shared, rgb_private, ir_private, _, _ = fcfs_split(
        rgb,
        ir,
        tau=args.tau,
        temp=args.temp,
        share_scale=args.share_scale,
        private_boost=args.private_boost,
    )

    plot_five_spectra(rgb, ir, shared, rgb_private, ir_private, Path(args.out))
    print(f"Saved visualization to {args.out}")


if __name__ == "__main__":
    main()

