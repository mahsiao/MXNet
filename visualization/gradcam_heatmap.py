import argparse
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
from visualization.feature_heatmap import (
    collect_sources,
    get_model_channels,
    get_pytorch_layers,
    get_pytorch_model,
    load_rgbt_tensor,
    overlay_heatmap,
    read_rgb_ir,
)

warnings.filterwarnings("ignore")


DEFAULT_LAYERS = [6, 16, 31]
DEFAULT_PROJECT = Path("visualization/runs/gradcam")
DEFAULT_LABEL_KIND = "fixlabel"


def normalize_heatmap(heatmap):
    heatmap = heatmap.detach().float().cpu().numpy()
    heatmap -= heatmap.min()
    heatmap /= heatmap.max() + 1e-6
    return (heatmap * 255).astype(np.uint8)


def image_to_label_path(image_path, label_kind=DEFAULT_LABEL_KIND):
    path = Path(image_path)
    parts = list(path.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = label_kind
            return Path(*parts).with_suffix(".txt")
    return path.parent.parent / label_kind / f"{path.stem}.txt"


def read_obb_labels(label_path):
    label_path = Path(label_path)
    if not label_path.exists():
        return []

    labels = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            if not values:
                continue
            cls = int(float(values[0]))
            coords = np.array([float(x) for x in values[1:]], dtype=np.float32)
            if coords.size >= 8:
                labels.append((cls, coords[:8].reshape(4, 2)))
    return labels


def gt_points_to_pixels(points, width, height):
    points = points.copy()
    if points.max() <= 1.01:
        points[:, 0] *= width
        points[:, 1] *= height
    return points


def gt_center_normalized(points):
    points = points.copy()
    if points.max() > 1.01:
        raise ValueError("GT-guided target expects normalized labels.")
    center = points.mean(axis=0)
    return float(center[0]), float(center[1])


def make_bbox_mask(labels, shape):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for _, points in labels:
        pts = gt_points_to_pixels(points, w, h).round().astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def scale_offsets(feature_shapes):
    offsets, start = {}, 0
    for name, h, w in feature_shapes:
        offsets[name] = start
        start += h * w
    return offsets


def select_gt_locations(labels, feature_shapes, gt_scale="P3", topk=1):
    if not labels or not feature_shapes:
        return []

    available = {name: (h, w) for name, h, w in feature_shapes}
    offsets = scale_offsets(feature_shapes)
    if gt_scale == "all":
        scales = list(available)
    elif gt_scale == "auto":
        scales = None
    else:
        scales = [gt_scale] if gt_scale in available else [feature_shapes[0][0]]

    locations = []
    for cls, points in labels:
        cx, cy = gt_center_normalized(points)
        if scales is None:
            xs = points[:, 0]
            ys = points[:, 1]
            size = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()))
            obj_scales = ["P3"] if size < 0.12 else (["P4"] if size < 0.24 else ["P5"])
            obj_scales = [s for s in obj_scales if s in available] or [feature_shapes[0][0]]
        else:
            obj_scales = scales

        for scale_name in obj_scales:
            h, w = available[scale_name]
            gx = min(max(cx * w, 0), w - 1)
            gy = min(max(cy * h, 0), h - 1)
            xs = np.arange(w, dtype=np.float32)
            ys = np.arange(h, dtype=np.float32)
            grid_x, grid_y = np.meshgrid(xs, ys)
            dist = (grid_x - gx) ** 2 + (grid_y - gy) ** 2
            nearest = np.argpartition(dist.reshape(-1), kth=min(topk, h * w) - 1)[:topk]
            for local_idx in nearest:
                locations.append((int(cls), offsets[scale_name] + int(local_idx), scale_name))
    return locations


def prediction_target(preds, nc, feature_shapes=None, gt_locations=None):
    """Use the strongest class response as the Grad-CAM target."""
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    if not isinstance(preds, torch.Tensor):
        raise RuntimeError(f"Unsupported prediction type for Grad-CAM: {type(preds)}")

    # OBB head layout is [xywh, cls..., angle], so class channels start at 4.
    class_scores = preds[:, 4 : 4 + nc, :]
    if gt_locations:
        valid_scores = []
        used = []
        for cls_idx, anchor_idx, scale_name in gt_locations:
            if 0 <= cls_idx < nc and 0 <= anchor_idx < class_scores.shape[-1]:
                valid_scores.append(class_scores[0, cls_idx, anchor_idx])
                used.append((cls_idx, anchor_idx, scale_name))
        if valid_scores:
            target = torch.stack(valid_scores).mean()
            info = {
                "class": used[0][0],
                "anchor": used[0][1],
                "scale": "+".join(sorted({u[2] for u in used})),
                "score": float(target.detach().cpu()),
                "mode": "gt",
                "locations": len(used),
            }
            return target, info

    flat_idx = int(class_scores.argmax().item())
    _, cls_idx, anchor_idx = np.unravel_index(flat_idx, tuple(class_scores.shape))

    scale_name = "unknown"
    if feature_shapes:
        offset = 0
        for name, h, w in feature_shapes:
            count = h * w
            if offset <= anchor_idx < offset + count:
                scale_name = name
                break
            offset += count

    info = {
        "class": int(cls_idx),
        "anchor": int(anchor_idx),
        "scale": scale_name,
        "score": float(class_scores.max().detach().cpu()),
        "mode": "max",
        "locations": 1,
    }
    return class_scores.max(), info


def gradcam_for_layer(torch_model, layers_seq, layer_id, image_tensor, nc, feature_shapes=None, gt_locations=None):
    activations = []

    def forward_hook(_, __, output):
        if isinstance(output, torch.Tensor):
            activations.append(output)
            output.retain_grad()

    handle = layers_seq[layer_id].register_forward_hook(forward_hook)
    try:
        image_tensor = image_tensor.detach().requires_grad_(True)
        torch_model.zero_grad(set_to_none=True)
        preds = torch_model(image_tensor)
        target, target_info = prediction_target(preds, nc, feature_shapes, gt_locations)
        target.backward()

        if not activations or activations[-1].grad is None:
            raise RuntimeError(f"No gradient captured for layer {layer_id}.")

        act = activations[-1][0]
        grad = activations[-1].grad[0]
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = (weights * act).sum(dim=0).relu()
        heatmap = normalize_heatmap(cam)
        stats = {
            "nonzero": float((heatmap > 0).mean()),
            "mean": float(heatmap.mean()),
            "max": int(heatmap.max()),
            "target": target_info,
        }
        return heatmap, stats
    finally:
        handle.remove()


def save_gradcam_outputs(heatmap, source, save_dir, layer_id, alpha, labels=None):
    layer_dir = Path(save_dir) / f"{layer_id}_gradcam"
    layer_dir.mkdir(parents=True, exist_ok=True)

    rgb, ir = read_rgb_ir(source)
    stem = Path(source).stem
    cv2.imwrite(str(layer_dir / f"{stem}_raw_gradcam.png"), heatmap)
    cv2.imwrite(str(layer_dir / f"{stem}_rgb_gradcam.png"), overlay_heatmap(rgb, heatmap, alpha))
    if ir is not None:
        cv2.imwrite(str(layer_dir / f"{stem}_ir_gradcam.png"), overlay_heatmap(ir, heatmap, alpha))

    if labels:
        mask = make_bbox_mask(labels, rgb.shape)
        heatmap_full = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        masked = cv2.bitwise_and(heatmap_full, heatmap_full, mask=mask)
        cv2.imwrite(str(layer_dir / f"{stem}_bbox_mask.png"), mask)
        cv2.imwrite(str(layer_dir / f"{stem}_raw_gradcam_bbox.png"), masked)
        cv2.imwrite(str(layer_dir / f"{stem}_rgb_gradcam_bbox.png"), overlay_heatmap(rgb, masked, alpha))
        if ir is not None:
            cv2.imwrite(str(layer_dir / f"{stem}_ir_gradcam_bbox.png"), overlay_heatmap(ir, masked, alpha))


def visualize_gradcam(
    model,
    source,
    layers=None,
    name="gradcam",
    project=DEFAULT_PROJECT,
    imgsz=640,
    alpha=0.45,
    device=None,
    gt_guided=True,
    label_kind=DEFAULT_LABEL_KIND,
    gt_scale="P3",
    topk=1,
):
    sources = collect_sources(source)
    if not sources:
        raise FileNotFoundError(f"No image sources found from: {source}")

    model = model if isinstance(model, DualYOLO) else DualYOLO(model)
    if get_model_channels(model, default=4) != 4:
        raise ValueError("Grad-CAM script currently expects a 4-channel RGBT checkpoint.")

    torch_model = get_pytorch_model(model)
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    torch_model.to(device).eval()
    for p in torch_model.parameters():
        p.requires_grad_(True)

    yaml = getattr(torch_model, "yaml", {}) or {}
    nc = int(yaml.get("nc", 5))
    layers_seq = get_pytorch_layers(model)
    layer_ids = DEFAULT_LAYERS if layers is None else layers
    save_dir = Path(project) / name

    for source_path in sources:
        image_tensor = load_rgbt_tensor(source_path, imgsz=imgsz, device=device)
        feature_shapes = []
        for scale_name, scale_layer in (("P3", 31), ("P4", 34), ("P5", 37)):
            if scale_layer < len(layers_seq):
                out_shape = None

                def shape_hook(_, __, output):
                    nonlocal out_shape
                    if isinstance(output, torch.Tensor):
                        out_shape = output.shape[-2:]

                handle = layers_seq[scale_layer].register_forward_hook(shape_hook)
                with torch.no_grad():
                    torch_model(image_tensor)
                handle.remove()
                if out_shape is not None:
                    feature_shapes.append((scale_name, int(out_shape[0]), int(out_shape[1])))

        label_path = image_to_label_path(source_path, label_kind)
        labels = read_obb_labels(label_path)
        gt_locations = select_gt_locations(labels, feature_shapes, gt_scale=gt_scale, topk=topk) if gt_guided else []
        if gt_guided and not gt_locations:
            print(f"No GT target found for {source_path}; fallback to max-score Grad-CAM.")

        for layer_id in layer_ids:
            if layer_id < 0 or layer_id >= len(layers_seq):
                raise IndexError(f"Layer {layer_id} is out of range. Model has {len(layers_seq)} layers.")
            heatmap, stats = gradcam_for_layer(
                torch_model, layers_seq, layer_id, image_tensor, nc, feature_shapes, gt_locations
            )
            save_gradcam_outputs(heatmap, source_path, save_dir, layer_id, alpha, labels=labels)
            target = stats["target"]
            print(
                f"{Path(source_path).name} layer={layer_id} target_mode={target['mode']} "
                f"target_cls={target['class']} target_scale={target['scale']} "
                f"locations={target['locations']} score={target['score']:.4f} "
                f"cam_mean={stats['mean']:.2f} cam_nonzero={stats['nonzero']:.3f}"
            )

    return save_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for RGBT OBB models.")
    parser.add_argument("--model", default=r"D:\CProject\MXNet\eval\ckpts\DV\DV-11n-mid.pt")
    parser.add_argument("--source", default=r"E:\CProject\Dataset\DVOBB\train\visible\images\00027.jpg")
    parser.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--name", default="dv_00027_gradcam")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-gt-guided", action="store_true", help="Use max-score Grad-CAM instead of GT-guided target.")
    parser.add_argument("--label-kind", default=DEFAULT_LABEL_KIND, help="Label folder name, e.g. fixlabel or labels.")
    parser.add_argument("--gt-scale", choices=["P3", "P4", "P5", "all", "auto"], default="P3")
    parser.add_argument("--topk", type=int, default=1, help="Nearest GT-center locations per GT and scale.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = visualize_gradcam(
        model=args.model,
        source=args.source,
        layers=args.layers,
        name=args.name,
        project=args.project,
        imgsz=args.imgsz,
        alpha=args.alpha,
        device=args.device,
        gt_guided=not args.no_gt_guided,
        label_kind=args.label_kind,
        gt_scale=args.gt_scale,
        topk=args.topk,
    )
    print(f"Grad-CAM heatmaps saved to: {out_dir}")
