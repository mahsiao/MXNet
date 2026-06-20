import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from prettytable import PrettyTable
from ultralytics.utils import YAML


DEFAULT_NAMES = ["car", "bus", "van", "truck", "freight_car"]
DEFAULT_IOU_THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]
COCO_SCALE_RANGES = {
    "small": (0.0, 32.0 * 32.0),
    "medium": (32.0 * 32.0, 96.0 * 96.0),
    "large": (96.0 * 96.0, float("inf")),
}
DEFAULT_DATA = Path(__file__).resolve().parent / "datasets" / "DVOBB.yaml"
DEFAULT_PRED = Path(__file__).resolve().parent / "runs" / "obb" / "DVOBB" / "yolo11-mid-p3-obb" / "predictions.json"
DEFAULT_PRED_ROOT = Path(__file__).resolve().parent / "runs"


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def signed_polygon_area(points):
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area * 0.5


def ensure_ccw(points):
    return points if signed_polygon_area(points) >= 0 else list(reversed(points))


def line_intersection(p1, p2, q1, q2):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return p2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return [px, py]


def inside(point, edge_start, edge_end):
    px, py = point
    ax, ay = edge_start
    bx, by = edge_end
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax) >= -1e-9


def polygon_clip(subject_polygon, clip_polygon):
    output = ensure_ccw(subject_polygon)
    clip_polygon = ensure_ccw(clip_polygon)

    for i, clip_start in enumerate(clip_polygon):
        clip_end = clip_polygon[(i + 1) % len(clip_polygon)]
        input_list = output
        output = []
        if not input_list:
            break

        prev_point = input_list[-1]
        for curr_point in input_list:
            curr_inside = inside(curr_point, clip_start, clip_end)
            prev_inside = inside(prev_point, clip_start, clip_end)
            if curr_inside:
                if not prev_inside:
                    output.append(line_intersection(prev_point, curr_point, clip_start, clip_end))
                output.append(curr_point)
            elif prev_inside:
                output.append(line_intersection(prev_point, curr_point, clip_start, clip_end))
            prev_point = curr_point

    return output


def clip_polygon_to_image(poly, width, height):
    image_poly = [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]]
    return polygon_clip(poly, image_poly)


def polygon_out_of_bounds(poly, width, height):
    return any(x < 0 or x > width or y < 0 or y > height for x, y in poly)


def oriented_iou(poly_a, poly_b):
    area_a = polygon_area(poly_a)
    area_b = polygon_area(poly_b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter_poly = polygon_clip(poly_a, poly_b)
    inter_area = polygon_area(inter_poly)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def xywhr_to_poly(rbox):
    cx, cy, w, h, angle = [float(x) for x in rbox[:5]]
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return [[cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a] for x, y in corners]


def bbox_to_poly(bbox):
    x, y, w, h = [float(v) for v in bbox[:4]]
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def parse_label_line(line, width, height, clip_mode="clip"):
    values = [float(x) for x in line.strip().split()]
    if len(values) < 9:
        return None, "skipped_malformed"
    cls = int(values[0])
    coords = values[1:9]
    normalized = max(coords) <= 1.01
    poly = []
    for i in range(0, 8, 2):
        x = coords[i] * width if normalized else coords[i]
        y = coords[i + 1] * height if normalized else coords[i + 1]
        poly.append([float(x), float(y)])

    clipped = False
    if polygon_out_of_bounds(poly, width, height):
        if clip_mode == "skip":
            return None, "skipped_out_of_bounds"
        if clip_mode == "clip":
            poly = clip_polygon_to_image(poly, width, height)
            clipped = True

    area = polygon_area(poly)
    if area <= 1e-6:
        return None, "skipped_invalid_area"
    return {"cls": cls, "poly": poly, "area": area}, "clipped" if clipped else "ok"


def labels_from_images_path(images_dir):
    parts = list(Path(images_dir).parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            return Path(*parts)
    return Path(images_dir).parent / "labels"


def resolve_dataset_paths(data_yaml, split):
    data_yaml = Path(data_yaml)
    data = YAML.load(data_yaml)
    root = Path(data.get("path", data_yaml.parent))
    images = Path(data[split])
    if not images.is_absolute():
        images = root / images
    labels = labels_from_images_path(images)
    names = data.get("names", DEFAULT_NAMES)
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    return images, labels, names


def load_ground_truth(images_dir, labels_dir, clip_mode="clip"):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    image_paths = sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )

    gts = []
    stats = defaultdict(int)
    for image_path in image_paths:
        with Image.open(image_path) as im:
            width, height = im.size

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parsed, status = parse_label_line(line, width, height, clip_mode=clip_mode)
            stats[status] += 1
            if parsed is None:
                continue
            parsed["image_id"] = image_path.stem
            gts.append(parsed)
    return gts, stats


def load_predictions(pred_json):
    preds = json.loads(Path(pred_json).read_text(encoding="utf-8"))
    parsed = []
    for pred in preds:
        if "poly" in pred:
            raw = [float(x) for x in pred["poly"]]
            poly = [[raw[i], raw[i + 1]] for i in range(0, len(raw), 2)]
        elif "rbox" in pred:
            poly = xywhr_to_poly(pred["rbox"])
        elif "bbox" in pred:
            poly = bbox_to_poly(pred["bbox"])
        else:
            continue

        image_id = Path(pred["file_name"]).stem if pred.get("file_name") else str(pred["image_id"])
        parsed.append(
            {
                "image_id": image_id,
                "cls": int(pred["category_id"]) - 1,
                "score": float(pred["score"]),
                "poly": poly,
            }
        )
    return parsed


def build_scale_ranges(gts, mode="coco"):
    if mode == "coco":
        return COCO_SCALE_RANGES

    areas = np.array([g["area"] for g in gts if g["area"] > 0], dtype=np.float64)
    if areas.size == 0:
        raise ValueError("No valid GT areas found for quantile scale split.")
    q33, q66 = np.quantile(areas, [1 / 3, 2 / 3])
    return {
        "small": (0.0, float(q33)),
        "medium": (float(q33), float(q66)),
        "large": (float(q66), float("inf")),
    }


def scale_name(area, scale_ranges):
    for name, (lo, hi) in scale_ranges.items():
        if lo <= area < hi:
            return name
    return "large"


def compute_ap(recalls, precisions):
    if len(recalls) == 0:
        return np.nan
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    recall_points = np.linspace(0, 1, 101)
    return float(np.mean([np.max(mpre[mrec >= r]) if np.any(mrec >= r) else 0.0 for r in recall_points]))


def evaluate_class_scale(gts, preds, cls, scale, iou_thr, scale_ranges):
    gt_filtered = [g for g in gts if g["cls"] == cls and scale_name(g["area"], scale_ranges) == scale]
    if not gt_filtered:
        return np.nan

    gt_by_image = defaultdict(list)
    for idx, gt in enumerate(gt_filtered):
        gt_item = dict(gt)
        gt_item["matched"] = False
        gt_item["id"] = idx
        gt_by_image[gt["image_id"]].append(gt_item)

    pred_filtered = sorted([p for p in preds if p["cls"] == cls], key=lambda x: x["score"], reverse=True)
    tp = np.zeros(len(pred_filtered), dtype=np.float32)
    fp = np.zeros(len(pred_filtered), dtype=np.float32)

    for i, pred in enumerate(pred_filtered):
        candidates = gt_by_image.get(pred["image_id"], [])
        best_iou = 0.0
        best_gt = None
        for gt in candidates:
            if gt["matched"]:
                continue
            iou = oriented_iou(pred["poly"], gt["poly"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        if best_gt is not None and best_iou >= iou_thr:
            tp[i] = 1
            best_gt["matched"] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(len(gt_filtered), 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return compute_ap(recalls, precisions)


def evaluate(gts, preds, names, iou_thresholds, scale_ranges):
    rows = []
    metrics = {}

    for scale in scale_ranges:
        cls_aps = []
        for cls, name in enumerate(names):
            ap_per_thr = [evaluate_class_scale(gts, preds, cls, scale, thr, scale_ranges) for thr in iou_thresholds]
            ap = float(np.nanmean(ap_per_thr)) if not np.all(np.isnan(ap_per_thr)) else np.nan
            ap50 = ap_per_thr[0]
            ap75 = ap_per_thr[5] if len(ap_per_thr) > 5 else np.nan
            count = sum(1 for g in gts if g["cls"] == cls and scale_name(g["area"], scale_ranges) == scale)
            if count:
                cls_aps.append(ap)
            rows.append(
                {
                    "scale": scale,
                    "class": name,
                    "instances": count,
                    "AP50": ap50,
                    "AP75": ap75,
                    "AP50-95": ap,
                }
            )
        metrics[f"OBB_AP_{scale}"] = float(np.nanmean(cls_aps)) if cls_aps else np.nan

    metrics["OBB_APs"] = metrics["OBB_AP_small"]
    metrics["OBB_APm"] = metrics["OBB_AP_medium"]
    metrics["OBB_APl"] = metrics["OBB_AP_large"]
    return rows, metrics


def print_table(rows, metrics):
    table = PrettyTable()
    table.title = "Scale-wise OBB AP"
    table.field_names = ["scale", "class", "instances", "AP50", "AP75", "AP50-95"]
    for row in rows:
        table.add_row(
            [
                row["scale"],
                row["class"],
                row["instances"],
                "nan" if np.isnan(row["AP50"]) else f"{row['AP50']:.4f}",
                "nan" if np.isnan(row["AP75"]) else f"{row['AP75']:.4f}",
                "nan" if np.isnan(row["AP50-95"]) else f"{row['AP50-95']:.4f}",
            ]
        )
    print(table)
    print(
        "Summary: "
        f"OBB_APs={metrics['OBB_APs']:.4f}, "
        f"OBB_APm={metrics['OBB_APm']:.4f}, "
        f"OBB_APl={metrics['OBB_APl']:.4f}"
    )


def save_outputs(rows, metrics, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "scale_obb_ap.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics_path = out_dir / "scale_obb_ap_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return csv_path, metrics_path


def find_latest_predictions(search_root=DEFAULT_PRED_ROOT):
    search_root = Path(search_root)
    candidates = sorted(
        search_root.rglob("predictions.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No predictions.json found under {search_root}. Run validation with save_json=True first."
        )
    return candidates[0]


def default_out_dir_for_prediction(pred_path):
    pred_path = Path(pred_path)
    return pred_path.parent / "scale_obb_ap"


def parse_args():
    parser = argparse.ArgumentParser(description="Scale-wise OBB AP using oriented polygon IoU.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset YAML. If provided, --images/--labels can be omitted.")
    parser.add_argument("--split", default="val", help="Dataset split key used with --data.")
    parser.add_argument("--images", help="Validation visible image directory.")
    parser.add_argument("--labels", help="Validation visible OBB label directory.")
    parser.add_argument("--pred", default=str(DEFAULT_PRED), help="OBB predictions.json generated by val(save_json=True).")
    parser.add_argument("--out-dir", help="Output directory. Defaults to <prediction_dir>/scale_obb_ap.")
    parser.add_argument("--names", nargs="+", default=DEFAULT_NAMES)
    parser.add_argument(
        "--scale-mode",
        choices=["coco", "quantile"],
        default="coco",
        help="Scale split mode. 'coco' uses 32^2/96^2; 'quantile' uses GT area 33%%/66%% quantiles.",
    )
    parser.add_argument(
        "--clip-labels",
        choices=["clip", "skip", "none"],
        default="clip",
        help="How to handle GT polygons outside image bounds.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pred_path = Path(args.pred)
    if not pred_path.exists() and str(pred_path) == str(DEFAULT_PRED):
        pred_path = find_latest_predictions()
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir_for_prediction(pred_path)

    if args.data:
        images, labels, names = resolve_dataset_paths(args.data, args.split)
    else:
        if not args.images or not args.labels:
            raise ValueError("Provide either --data or both --images and --labels.")
        images, labels, names = Path(args.images), Path(args.labels), args.names

    gts, gt_stats = load_ground_truth(images, labels, clip_mode=args.clip_labels)
    preds = load_predictions(pred_path)
    scale_ranges = build_scale_ranges(gts, mode=args.scale_mode)
    rows, metrics = evaluate(gts, preds, names, DEFAULT_IOU_THRESHOLDS, scale_ranges)
    print_table(rows, metrics)
    csv_path, metrics_path = save_outputs(rows, metrics, out_dir)
    print(f"Images: {images}")
    print(f"Labels: {labels}")
    print(f"Predictions: {pred_path}")
    print(f"Scale mode: {args.scale_mode}")
    print(f"Scale ranges: {scale_ranges}")
    print(f"GT label handling: {dict(gt_stats)}")
    print(f"CSV saved to: {csv_path}")
    print(f"Metrics saved to: {metrics_path}")
