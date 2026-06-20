import glob
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import DualYOLO
from ultralytics.data.augment import LetterBox

warnings.filterwarnings("ignore")


DEFAULT_LAYERS = {
    6: "rgb_p3",
    16: "ir_p3",
    21: "fusion_p3",
}
DEFAULT_PROJECT = Path("/heatmap")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def collect_sources(source):
    if isinstance(source, (list, tuple)):
        paths = []
        for item in source:
            paths.extend(collect_sources(item))
        return paths

    source = str(source)
    if any(ch in source for ch in "*?[]"):
        return [Path(p) for p in sorted(glob.glob(source, recursive=True)) if Path(p).suffix.lower() in IMAGE_SUFFIXES]

    path = Path(source)
    if path.is_dir():
        return [p for p in sorted(path.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    return []


def get_pytorch_layers(model):
    """Return the internal Sequential layer list used by the PyTorch backend."""
    pytorch_model = get_pytorch_model(model)
    layers = getattr(pytorch_model, "model", None)
    if layers is None:
        raise RuntimeError("Unable to locate PyTorch model layers for hook registration.")
    return layers


def get_model_channels(model, default=4):
    pytorch_model = get_pytorch_model(model)
    yaml = getattr(pytorch_model, "yaml", {}) or {}
    return int(yaml.get("channels", yaml.get("ch", default)))


def get_pytorch_model(model):
    predictor = getattr(model, "predictor", None)
    backend = getattr(predictor, "model", None) if predictor is not None else None
    if backend is None:
        return getattr(model, "model", model)
    return getattr(backend, "model", backend)


def tensor_from_hook_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            tensor = tensor_from_hook_output(item)
            if tensor is not None:
                return tensor
    return None


def register_feature_hooks(layers, layer_ids):
    features = {layer_id: [] for layer_id in layer_ids}
    handles = []

    def make_hook(layer_id):
        def hook(_, __, output):
            tensor = tensor_from_hook_output(output)
            if tensor is not None and tensor.ndim == 4:
                features[layer_id].append(tensor.detach().float().cpu())

        return hook

    for layer_id in layer_ids:
        if layer_id < 0 or layer_id >= len(layers):
            raise IndexError(f"Layer {layer_id} is out of range. Model has {len(layers)} layers.")
        handles.append(layers[layer_id].register_forward_hook(make_hook(layer_id)))
    return features, handles


def feature_to_heatmap(feature, mode="mean_abs"):
    """Convert one feature tensor CxHxW to a normalized 0-255 heatmap."""
    if mode == "mean":
        heatmap = feature.mean(dim=0)
    elif mode == "max":
        heatmap = feature.max(dim=0).values
    elif mode == "mean_abs":
        heatmap = feature.abs().mean(dim=0)
    else:
        raise ValueError(f"Unsupported heatmap mode: {mode}")

    heatmap = heatmap.numpy()
    heatmap -= heatmap.min()
    heatmap /= heatmap.max() + 1e-6
    return (heatmap * 255).astype(np.uint8)


def overlay_heatmap(image, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)


def read_rgb_ir(source):
    rgb = cv2.imread(str(source))
    ir_path = Path(str(source).replace("visible", "infrared"))
    ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
    if rgb is None:
        raise FileNotFoundError(f"RGB image not found: {source}")
    ir_bgr = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR) if ir is not None else None
    return rgb, ir_bgr


def load_rgbt_tensor(source, imgsz, device):
    rgb = cv2.imread(str(source))
    ir_path = Path(str(source).replace("visible", "infrared"))
    ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
    if rgb is None:
        raise FileNotFoundError(f"RGB image not found: {source}")
    if ir is None:
        raise FileNotFoundError(f"IR image not found: {ir_path}")

    if rgb.shape[:2] != ir.shape[:2]:
        ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)

    b, g, r = cv2.split(rgb)
    rgbt = cv2.merge((b, g, r, ir))
    rgbt = LetterBox((imgsz, imgsz), auto=True, stride=32)(image=rgbt)

    chw = rgbt.transpose(2, 0, 1)
    img3c = chw[:3][::-1]
    img1c = chw[-1:]
    tensor = np.ascontiguousarray(np.concatenate((img3c, img1c), axis=0))
    tensor = torch.from_numpy(tensor).unsqueeze(0).to(device).float() / 255.0
    return tensor


def save_layer_heatmaps(features, sources, save_dir, layer_names, mode="mean_abs", alpha=0.45):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for layer_id, batches in features.items():
        if not batches:
            print(f"No feature captured for layer {layer_id}.")
            continue

        layer_features = torch.cat(batches, dim=0)
        layer_name = layer_names.get(layer_id, f"layer{layer_id}")
        layer_dir = save_dir / f"{layer_id}_{layer_name}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        for i, source in enumerate(sources):
            if i >= len(layer_features):
                break

            rgb, ir = read_rgb_ir(source)
            heatmap = feature_to_heatmap(layer_features[i], mode=mode)
            cv2.imwrite(str(layer_dir / f"{Path(source).stem}_rgb_heatmap.png"), overlay_heatmap(rgb, heatmap, alpha))
            if ir is not None:
                cv2.imwrite(str(layer_dir / f"{Path(source).stem}_ir_heatmap.png"), overlay_heatmap(ir, heatmap, alpha))
            cv2.imwrite(str(layer_dir / f"{Path(source).stem}_raw_heatmap.png"), heatmap)


def visualize_feature_heatmaps(
    model,
    source,
    layers=None,
    name="yolo11_mid_p3",
    project=DEFAULT_PROJECT,
    imgsz=640,
    mode="mean_abs",
    alpha=0.45,
    device=None,
    **predict_kwargs,
):
    """Save channel-aggregated feature response heatmaps for selected model layers.

    Default layers for mxrecode/baseline/yolo11-mid-p3.yaml:
        6  = RGB branch P3
        12 = IR branch P3
        14 = fused P3 after 1x1 Conv
    """
    layer_names = DEFAULT_LAYERS if layers is None else {layer: f"layer{layer}" for layer in layers}
    layer_ids = list(layer_names)
    sources = collect_sources(source)
    if not sources:
        raise FileNotFoundError(f"No image sources found from: {source}")

    model = model if isinstance(model, DualYOLO) else DualYOLO(model)
    model_channels = get_model_channels(model, default=4)
    if model_channels != 4:
        raise ValueError(
            f"Checkpoint expects {model_channels} input channel(s), but RGBT heatmap visualization requires 4. "
            "Please use a RGBT checkpoint or change the script to the matching single-modal loader."
        )

    torch_model = get_pytorch_model(model)
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    torch_model.to(device).eval()
    layers_seq = get_pytorch_layers(model)
    features, handles = register_feature_hooks(layers_seq, layer_ids)

    try:
        with torch.inference_mode():
            for source_path in sources:
                im = load_rgbt_tensor(source_path, imgsz=imgsz, device=device)
                torch_model(im)
    finally:
        for handle in handles:
            handle.remove()

    save_dir = Path(project) / name
    save_layer_heatmaps(features, sources, save_dir, layer_names, mode=mode, alpha=alpha)
    return save_dir


if __name__ == "__main__":
    ckpt = r"D:\CProject\MXNet\eval\ckpts\DV\DV-11n-mid.pt"
    source = r"E:\CProject\Dataset\DVOBB\train\visible\images\00027.jpg"
    out_dir = visualize_feature_heatmaps(
        model=ckpt,
        source=source,
        layers=[6, 16, 21, 8, 18, 22, 10, 20, 23],
        name="dv_00027_mid_p3_layers_6_12_21",
        imgsz=640,
        mode="mean_abs",
        alpha=0.45,
    )
    print(f"Heatmaps saved to: {out_dir}")
