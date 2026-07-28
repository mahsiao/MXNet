import argparse
import csv
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


DEFAULT_VISIBLE_DIR = Path(r"E:\CProject\Dataset\DVOBB\train\visible\images")
DEFAULT_INFRARED_DIR = Path(r"E:\CProject\Dataset\DVOBB\train\infrared\images")
DEFAULT_IMAGE_NAME = "00090.jpg"
DEFAULT_OUTPUT_DIR = ROOT / "visualization" / "fcfs90"
DEFAULT_CKPT = Path(r"D:\CProject\MXNet\eval\ckpts\DV\DV-11n-mid-fcfs-private-residual.pt")

COMPONENT_KEYS = ["rgb", "ir", "shared", "rgb_private", "ir_private"]
COMPONENT_TITLES = ["RGB", "IR", "Shared", "RGB-Private", "IR-Private"]


def resolve_pair_paths(args):
    """Resolve paired RGB/IR paths from either explicit paths or one shared image name."""
    image_name = args.image_name
    rgb_path = Path(args.rgb) if args.rgb else Path(args.visible_dir) / image_name
    ir_path = Path(args.ir) if args.ir else Path(args.infrared_dir) / image_name
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    if not ir_path.exists():
        raise FileNotFoundError(f"IR image not found: {ir_path}")
    return rgb_path, ir_path


def level_name(index):
    """Map FCFS layer index to a friendly feature level name."""
    return f"P{index + 3}" if index < 3 else f"L{index}"


def parse_levels(levels_text, num_layers):
    """Parse a comma-separated level list such as p3,p4,p5,all into layer indices."""
    text = levels_text.strip().lower()
    if text == "all":
        return list(range(min(3, num_layers)))

    mapping = {"p3": 0, "p4": 1, "p5": 2}
    indices = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in mapping:
            raise ValueError(f"Unsupported level '{item}'. Use p3, p4, p5, p3,p4, or all.")
        index = mapping[item]
        if index >= num_layers:
            raise ValueError(f"Requested {item.upper()}, but the model only has {num_layers} FCFS layer(s).")
        if index not in indices:
            indices.append(index)
    if not indices:
        raise ValueError("No valid levels were requested.")
    return indices


def load_rgb_ir_tensor(rgb_path, ir_path, size=640, device="cpu"):
    """Load paired RGB/IR images as one 4-channel YOLO input tensor [1,4,H,W]."""
    rgb = Image.open(rgb_path).convert("RGB").resize((size, size), Image.BILINEAR)
    ir = Image.open(ir_path).convert("L").resize((size, size), Image.BILINEAR)

    rgb_arr = np.asarray(rgb).astype(np.float32) / 255.0
    ir_arr = np.asarray(ir).astype(np.float32) / 255.0

    rgb_tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1)
    ir_tensor = torch.from_numpy(ir_arr)[None]
    x = torch.cat([rgb_tensor, ir_tensor], dim=0)[None]
    return x.to(device)


def find_fcfs_layers(model):
    """Find FCFS-style layers in a YOLO model."""
    target_types = (FCFS, FCFSShared, FCFSPrivateResidual)
    return [(name, module) for name, module in model.named_modules() if isinstance(module, target_types)]


def capture_fcfs_features(model, image_tensor, fcfs_index=0):
    """
    Forward once and capture the input features of the selected FCFS layer.

    The selected layer may output only shared/fused features, so the script
    recomputes shared/RGB-private/IR-private and the internal frequency masks
    from the layer input features for visualization.
    """
    layers = find_fcfs_layers(model)
    if not layers:
        raise RuntimeError("No FCFS/FCFSShared/FCFSPrivateResidual layer found in the loaded model.")
    if fcfs_index < 0 or fcfs_index >= len(layers):
        names = ", ".join(f"{i}:{name}" for i, (name, _) in enumerate(layers))
        raise IndexError(f"fcfs_index={fcfs_index} is out of range. Available FCFS layers: {names}")

    layer_name, layer = layers[fcfs_index]
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
        raise RuntimeError(f"Failed to capture features from FCFS layer {layer_name}.")

    with torch.no_grad():
        freq_components = compute_fcfs_frequency_components(captured["rgb"], captured["ir"], layer)

    return {
        "layer_name": layer_name,
        "spatial_size": captured["rgb"].shape[-2:],
        **freq_components,
    }


def compute_fcfs_frequency_components(x, y, layer):
    """Compute FCFS internal frequency-domain components and masks."""
    eps = layer.eps
    spatial_size = x.shape[-2:]

    with torch.cuda.amp.autocast(enabled=False):
        x = x.float()
        y = y.float()

        X = torch.fft.rfft2(x, norm="ortho")
        Y = torch.fft.rfft2(y, norm="ortho")

        Sxx = (X.real ** 2 + X.imag ** 2) + eps
        Syy = (Y.real ** 2 + Y.imag ** 2) + eps
        Sxy = X * torch.conj(Y)

        coh = (Sxy.real ** 2 + Sxy.imag ** 2) / (Sxx * Syy)
        strength_balance = torch.sqrt(Sxx * Syy) / (Sxx + Syy + eps)

        Mc = torch.sigmoid(((coh * strength_balance) - layer.tau) / layer.temp)
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

        Zc_hat = Mc * (X + Y)
        # Zc_hat = Mc * (X + Y) * 0.5
        Zr_hat = Mr * X
        Zi_hat = Mi * Y

        # Reconstruct spatial outputs too, so stats can compare feature energy.
        Zc = torch.fft.irfft2(Zc_hat, s=spatial_size, norm="ortho")
        Zr = torch.fft.irfft2(Zr_hat, s=spatial_size, norm="ortho")
        Zi = torch.fft.irfft2(Zi_hat, s=spatial_size, norm="ortho")

    return {
        "rgb": x.detach(),
        "ir": y.detach(),
        "shared": Zc.detach(),
        "rgb_private": Zr.detach(),
        "ir_private": Zi.detach(),
        "rgb_hat": X.detach(),
        "ir_hat": Y.detach(),
        "shared_hat": Zc_hat.detach(),
        "rgb_private_hat": Zr_hat.detach(),
        "ir_private_hat": Zi_hat.detach(),
        "Mc": Mc.detach(),
        "Mr": Mr.detach(),
        "Mi": Mi.detach(),
        "coherence": coh.detach(),
        "strength_balance": strength_balance.detach(),
        "diff_mag": diff_mag.detach(),
    }


def rfft_component_to_full_fft(component, spatial_size):
    """Convert an FCFS rfft2 component into a full fft2 spectrum for square plots."""
    component = component.detach().cpu()
    spatial = torch.fft.irfft2(component, s=spatial_size, norm="ortho")
    return torch.fft.fft2(spatial, norm="ortho")


def tensor_to_full_fft(tensor):
    """Compute full fft2 spectrum from a spatial feature tensor."""
    tensor = tensor.detach().float().cpu()
    return torch.fft.fft2(tensor, norm="ortho")


def reduce_channel_maps(maps, channel_mode="mean", topk=16):
    """Reduce [B,C,H,W] frequency maps to one [H,W] visualization map."""
    maps = maps.reshape(-1, *maps.shape[-2:])
    if channel_mode == "mean":
        return maps.mean(dim=0)
    if channel_mode == "max":
        return maps.max(dim=0).values
    if channel_mode == "topk":
        k = min(max(int(topk), 1), maps.shape[0])
        scores = maps.flatten(1).sum(dim=1)
        indices = torch.topk(scores, k=k, largest=True).indices
        return maps[indices].mean(dim=0)
    raise ValueError(f"Unsupported channel_mode: {channel_mode}")


def spectrum_log_map(full_fft, channel_mode="mean", topk=16):
    """Convert channel spectra into one log-magnitude map with low frequency centered."""
    mag = torch.log1p(torch.abs(full_fft))
    mag = torch.fft.fftshift(mag, dim=(-2, -1))
    return reduce_channel_maps(mag, channel_mode=channel_mode, topk=topk).numpy()


def spectrum_power_map(full_fft, channel_mode="mean", topk=16):
    """Convert channel power spectra into one map with low frequency centered."""
    power = torch.abs(full_fft) ** 2
    power = torch.fft.fftshift(power, dim=(-2, -1))
    return reduce_channel_maps(power, channel_mode=channel_mode, topk=topk).numpy()


def get_full_ffts(features, source="fcfs"):
    """
    Return full square FFTs for RGB/IR/shared/private components.

    source='fcfs' uses the FCFS internal X/Y/Z_hat components and reconstructs
    their full Hermitian spectra. This is closest to paper-style frequency-domain
    visualization while avoiding rfft2 half-plane rectangles.
    source='spatial' computes fft2 after irfft2 spatial reconstruction.
    """
    if source == "fcfs":
        spatial_size = features["spatial_size"]
        return {
            "rgb": rfft_component_to_full_fft(features["rgb_hat"], spatial_size),
            "ir": rfft_component_to_full_fft(features["ir_hat"], spatial_size),
            "shared": rfft_component_to_full_fft(features["shared_hat"], spatial_size),
            "rgb_private": rfft_component_to_full_fft(features["rgb_private_hat"], spatial_size),
            "ir_private": rfft_component_to_full_fft(features["ir_private_hat"], spatial_size),
        }
    if source == "spatial":
        return {key: tensor_to_full_fft(features[key]) for key in COMPONENT_KEYS}
    raise ValueError(f"Unsupported source: {source}")


def normalize_shared(maps, p_low=1, p_high=99):
    """Normalize all maps with one shared range for fair side-by-side comparison."""
    stacked = np.stack(maps, axis=0)
    lo, hi = np.percentile(stacked, [p_low, p_high])
    return [np.clip((m - lo) / (hi - lo + 1e-8), 0.0, 1.0) for m in maps]


def normalize_each(maps, p_low=1, p_high=99):
    """Normalize each map independently to reveal weak but structured spectra."""
    normalized = []
    for m in maps:
        lo, hi = np.percentile(m, [p_low, p_high])
        normalized.append(np.clip((m - lo) / (hi - lo + 1e-8), 0.0, 1.0))
    return normalized


def gaussian_kernel1d(sigma):
    """Build a small Gaussian kernel for optional display-only smoothing."""
    if sigma <= 0:
        return None
    radius = int(3 * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_smooth2d(image, sigma):
    """Lightly smooth one spectrum map for display only."""
    kernel = gaussian_kernel1d(sigma)
    if kernel is None:
        return image
    pad = len(kernel) // 2
    padded = np.pad(image, ((0, 0), (pad, pad)), mode="reflect")
    smoothed = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
    padded = np.pad(smoothed, ((pad, pad), (0, 0)), mode="reflect")
    return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, padded)


def plot_five_spectra(maps, output, title, shared_scale=True, smooth_sigma=0.0):
    """Save five-column frequency visualization with a Turbo colormap."""
    maps_to_plot = [gaussian_smooth2d(m, smooth_sigma) for m in maps]
    maps_to_plot = normalize_shared(maps_to_plot) if shared_scale else normalize_each(maps_to_plot)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), constrained_layout=True)
    for ax, name, spec in zip(axes, COMPONENT_TITLES, maps_to_plot):
        im = ax.imshow(spec, cmap="turbo", vmin=0.0, vmax=1.0)
        ax.set_title(name, fontsize=11)
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.suptitle(title, fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def radial_profile(power_map, bins=64):
    """Compute radial mean power profile from a centered square/rectangular spectrum."""
    h, w = power_map.shape
    yy, xx = np.indices((h, w))
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = radius / (radius.max() + 1e-8)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(radius.ravel(), bin_edges) - 1
    values = power_map.ravel()
    profile = np.zeros(bins, dtype=np.float64)
    for i in range(bins):
        mask = bin_ids == i
        profile[i] = values[mask].mean() if np.any(mask) else 0.0
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centers, profile


def plot_radial_profiles(power_maps, output, title):
    """Save radial frequency-energy curves for the five components."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for key, label in zip(COMPONENT_KEYS, COMPONENT_TITLES):
        r, profile = radial_profile(power_maps[key])
        ax.plot(r, np.log1p(profile), label=label, linewidth=1.8)
    ax.set_xlabel("Normalized frequency radius")
    ax.set_ylabel("log(1 + mean power)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def band_energy(power_map):
    """Return low/mid/high frequency energy ratios from a centered power map."""
    h, w = power_map.shape
    yy, xx = np.indices((h, w))
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = radius / (radius.max() + 1e-8)
    total = float(power_map.sum() + 1e-12)
    low = float(power_map[radius < 0.25].sum()) / total
    mid = float(power_map[(radius >= 0.25) & (radius < 0.60)].sum()) / total
    high = float(power_map[radius >= 0.60].sum()) / total
    return low, mid, high, total


def plot_band_energy(power_maps, output, title):
    """Save stacked low/mid/high frequency energy ratio bars."""
    lows, mids, highs = [], [], []
    for key in COMPONENT_KEYS:
        low, mid, high, _ = band_energy(power_maps[key])
        lows.append(low)
        mids.append(mid)
        highs.append(high)

    x = np.arange(len(COMPONENT_KEYS))
    fig, ax = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
    ax.bar(x, lows, label="Low < 0.25r")
    ax.bar(x, mids, bottom=lows, label="Mid 0.25-0.60r")
    ax.bar(x, highs, bottom=np.array(lows) + np.array(mids), label="High >= 0.60r")
    ax.set_xticks(x)
    ax.set_xticklabels(COMPONENT_TITLES, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Energy ratio")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mask_histograms(features, output, title):
    """Save histograms for FCFS masks and coherence-related quantities."""
    items = [
        ("Mc shared mask", features["Mc"]),
        ("Mr RGB-private mask", features["Mr"]),
        ("Mi IR-private mask", features["Mi"]),
        ("Coherence", features["coherence"]),
        ("Strength balance", features["strength_balance"]),
        ("Difference magnitude", features["diff_mag"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6), constrained_layout=True)
    for ax, (name, tensor) in zip(axes.ravel(), items):
        values = tensor.detach().float().cpu().numpy().ravel()
        values = values[np.isfinite(values)]
        ax.hist(values, bins=80, range=(0, 1), color="#2f6f9f", alpha=0.85)
        ax.set_title(name, fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.2)
    fig.suptitle(title, fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_stats_csv(features, full_ffts, power_maps, output):
    """Write numerical spectrum and spatial-feature statistics for later analysis."""
    rows = []
    for key, title in zip(COMPONENT_KEYS, COMPONENT_TITLES):
        tensor = features[key].detach().float().cpu()
        fft_abs = torch.abs(full_ffts[key])
        low, mid, high, total_power = band_energy(power_maps[key])
        rows.append({
            "component": title,
            "spatial_mean_abs": float(tensor.abs().mean()),
            "spatial_rms": float(torch.sqrt(torch.mean(tensor ** 2))),
            "fft_mean_magnitude": float(fft_abs.mean()),
            "fft_rms_magnitude": float(torch.sqrt(torch.mean(fft_abs ** 2))),
            "fft_total_power": total_power,
            "low_energy_ratio": low,
            "mid_energy_ratio": mid,
            "high_energy_ratio": high,
        })

    for key in ["Mc", "Mr", "Mi", "coherence", "strength_balance", "diff_mag"]:
        values = features[key].detach().float().cpu()
        rows.append({
            "component": key,
            "spatial_mean_abs": "",
            "spatial_rms": "",
            "fft_mean_magnitude": float(values.mean()),
            "fft_rms_magnitude": float(torch.sqrt(torch.mean(values ** 2))),
            "fft_total_power": float(values.sum()),
            "low_energy_ratio": "",
            "mid_energy_ratio": "",
            "high_energy_ratio": "",
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_analysis_pack(
    features,
    output_dir,
    stem,
    level,
    source="fcfs",
    channel_mode="mean",
    topk=16,
    smooth_sigma=0.0,
):
    """Save paper-style spectra, curves, histograms, and CSV stats for one FCFS level."""
    full_ffts = get_full_ffts(features, source=source)
    log_maps = [spectrum_log_map(full_ffts[key], channel_mode=channel_mode, topk=topk) for key in COMPONENT_KEYS]
    power_maps = {
        key: spectrum_power_map(full_ffts[key], channel_mode=channel_mode, topk=topk)
        for key in COMPONENT_KEYS
    }

    prefix = output_dir / channel_mode / f"fcfs_{stem}_{source}_{channel_mode}_{level.lower()}"
    layer_title = f"{level} {features['layer_name']}, channel={channel_mode}"
    plot_five_spectra(
        log_maps,
        prefix.with_name(prefix.name + "_spectra_shared_scale.png"),
        f"FCFS frequency spectra, shared Turbo scale ({layer_title})",
        shared_scale=True,
        smooth_sigma=smooth_sigma,
    )
    plot_five_spectra(
        log_maps,
        prefix.with_name(prefix.name + "_spectra_self_norm.png"),
        f"FCFS frequency spectra, independent normalization ({layer_title})",
        shared_scale=False,
        smooth_sigma=smooth_sigma,
    )
    plot_radial_profiles(
        power_maps,
        prefix.with_name(prefix.name + "_radial_profiles.png"),
        f"Radial frequency energy profiles ({layer_title})",
    )
    plot_band_energy(
        power_maps,
        prefix.with_name(prefix.name + "_band_energy.png"),
        f"Low/mid/high frequency energy ratios ({layer_title})",
    )
    plot_mask_histograms(
        features,
        prefix.with_name(prefix.name + "_mask_histograms.png"),
        f"FCFS mask and coherence distributions ({layer_title})",
    )
    write_stats_csv(
        features,
        full_ffts,
        power_maps,
        prefix.with_name(prefix.name + "_stats.csv"),
    )
    return prefix


def main():
    parser = argparse.ArgumentParser(description="Visualize true FCFS feature-domain spectra from a trained YOLO checkpoint.")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="Path to a trained FCFS checkpoint.")
    parser.add_argument("--image-name", default=DEFAULT_IMAGE_NAME, help="Paired RGB/IR image file name, e.g. 00003.jpg.")
    parser.add_argument("--visible-dir", default=str(DEFAULT_VISIBLE_DIR), help="Directory containing visible RGB images.")
    parser.add_argument("--infrared-dir", default=str(DEFAULT_INFRARED_DIR), help="Directory containing infrared images.")
    parser.add_argument("--rgb", default=None, help="Optional explicit RGB image path. If omitted, visible-dir/image-name is used.")
    parser.add_argument("--ir", default=None, help="Optional explicit IR image path. If omitted, infrared-dir/image-name is used.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for output analysis figures.")
    parser.add_argument("--size", type=int, default=640, help="Resize RGB/IR inputs to size x size.")
    parser.add_argument("--fcfs-index", type=int, default=0, help="Which FCFS layer to visualize in single-layer mode.")
    parser.add_argument(
        "--source",
        choices=["fcfs", "spatial"],
        default="fcfs",
        help="fcfs reconstructs full spectra from internal X/Y/Z_hat; spatial draws fft2 of reconstructed spatial features.",
    )
    parser.add_argument(
        "--channel-mode",
        choices=["mean", "topk", "max"],
        default="mean",
        help="How to reduce feature channels: mean is objective, topk is paper-display friendly, max shows strongest responses.",
    )
    parser.add_argument("--topk", type=int, default=16, help="Number of channels used when --channel-mode topk.")
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.0,
        help="Optional display-only Gaussian smoothing for spectrum images. Try 0.3 or 0.5 first.",
    )
    parser.add_argument(
        "--levels",
        default="all",
        help="Feature levels to visualize in default mode: p3, p4, p5, p3,p4, or all.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--list-layers", action="store_true", help="List FCFS layers in the checkpoint and exit.")
    parser.add_argument("--single-layer", action="store_true", help="Only visualize fcfs-index instead of default P3/P4/P5.")
    args = parser.parse_args()

    device = torch.device(args.device)
    yolo = YOLO(args.ckpt)
    model = yolo.model.to(device)

    layers = find_fcfs_layers(model)
    if args.list_layers:
        for i, (name, module) in enumerate(layers):
            print(f"{i}: {name} ({module.__class__.__name__})")
        return

    rgb_path, ir_path = resolve_pair_paths(args)
    image_tensor = load_rgb_ir_tensor(rgb_path, ir_path, size=args.size, device=device)
    output_dir = Path(args.out_dir) / Path(args.image_name).stem
    stem = Path(args.image_name).stem

    if args.single_layer:
        level = level_name(args.fcfs_index)
        features = capture_fcfs_features(model, image_tensor, fcfs_index=args.fcfs_index)
        prefix = save_analysis_pack(
            features,
            output_dir,
            stem,
            level,
            source=args.source,
            channel_mode=args.channel_mode,
            topk=args.topk,
            smooth_sigma=args.smooth_sigma,
        )
        print(f"Visualized layer: {args.fcfs_index}:{features['layer_name']}")
        print(f"Spectrum source: {args.source}")
        print(f"Channel mode: {args.channel_mode}")
        print(f"Saved analysis pack with prefix: {prefix}")
        return

    for index in parse_levels(args.levels, len(layers)):
        level = level_name(index)
        features = capture_fcfs_features(model, image_tensor, fcfs_index=index)
        prefix = save_analysis_pack(
            features,
            output_dir,
            stem,
            level,
            source=args.source,
            channel_mode=args.channel_mode,
            topk=args.topk,
            smooth_sigma=args.smooth_sigma,
        )
        print(f"Visualized layer: {index}:{features['layer_name']} -> {prefix}*")


if __name__ == "__main__":
    main()

