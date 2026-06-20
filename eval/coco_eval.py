import argparse
import json
import math
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


DEFAULT_NAMES = ["car", "bus", "van", "truck", "freight_car"]


def polygon_area(points):
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def parse_yolo_label(line, width, height):
    values = [float(x) for x in line.strip().split()]
    if len(values) < 5:
        return None

    cls = int(values[0])
    coords = values[1:]

    if len(coords) == 8:
        normalized = max(coords) <= 1.01
        points = []
        for i in range(0, 8, 2):
            x = coords[i] * width if normalized else coords[i]
            y = coords[i + 1] * height if normalized else coords[i + 1]
            points.append([float(x), float(y)])
        bbox = polygon_to_bbox(points)
        area = polygon_area(points)
        segmentation = [[v for p in points for v in p]]
        return cls, bbox, area, segmentation

    if len(coords) == 4:
        x, y, w, h = coords
        if max(coords) <= 1.01:
            x, w = x * width, w * width
            y, h = y * height, h * height
        bbox = [x - w / 2, y - h / 2, w, h]
        area = max(0.0, w) * max(0.0, h)
        return cls, bbox, area, []

    return None


def build_coco_gt(images_dir, labels_dir, names, out_json, area_mode="polygon"):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    image_paths = sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(names)],
    }
    ann_id = 1

    for image_path in image_paths:
        with Image.open(image_path) as im:
            width, height = im.size

        image_id = image_path.stem
        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parsed = parse_yolo_label(line, width, height)
            if parsed is None:
                continue
            cls, bbox, poly_area, segmentation = parsed
            bbox_area = bbox[2] * bbox[3]
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls + 1,
                    "bbox": [round(float(x), 3) for x in bbox],
                    "area": round(float(poly_area if area_mode == "polygon" else bbox_area), 3),
                    "segmentation": segmentation,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    return out_json


def convert_predictions_to_bbox(pred_json, out_json):
    pred_json = Path(pred_json)
    preds = json.loads(pred_json.read_text(encoding="utf-8"))
    converted = []

    for pred in preds:
        item = {
            "image_id": pred["image_id"],
            "category_id": int(pred["category_id"]),
            "score": float(pred["score"]),
        }

        if "bbox" in pred:
            item["bbox"] = [float(x) for x in pred["bbox"]]
        elif "poly" in pred:
            poly = [float(x) for x in pred["poly"]]
            points = [[poly[i], poly[i + 1]] for i in range(0, len(poly), 2)]
            item["bbox"] = polygon_to_bbox(points)
        elif "rbox" in pred:
            # Fallback for [cx, cy, w, h, angle]. Prefer "poly" when available.
            cx, cy, w, h = [float(x) for x in pred["rbox"][:4]]
            item["bbox"] = [cx - w / 2, cy - h / 2, w, h]
        else:
            continue

        if not all(math.isfinite(x) for x in item["bbox"]):
            continue
        converted.append(item)

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(converted, ensure_ascii=False), encoding="utf-8")
    return out_json


def evaluate_coco(gt_json, pred_json):
    coco_gt = COCO(str(gt_json))
    coco_dt = coco_gt.loadRes(str(pred_json))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    names = [
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR1",
        "AR10",
        "AR100",
        "ARs",
        "ARm",
        "ARl",
    ]
    return dict(zip(names, coco_eval.stats.tolist()))


def parse_args():
    parser = argparse.ArgumentParser(description="COCO bbox APs/APm/APl evaluation for DroneVehicle YOLO/OBB results.")
    parser.add_argument("--images", required=True, help="Validation visible image directory.")
    parser.add_argument("--labels", required=True, help="Validation visible label directory in YOLO OBB format.")
    parser.add_argument("--pred", required=True, help="predictions.json generated by val(save_json=True).")
    parser.add_argument("--out-dir", default="eval/runs/coco_eval")
    parser.add_argument("--names", nargs="+", default=DEFAULT_NAMES)
    parser.add_argument(
        "--area-mode",
        choices=["polygon", "bbox"],
        default="polygon",
        help="Area used for APs/APm/APl grouping. polygon is closer to OBB object area; bbox matches HBB area.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out_dir)
    gt_json = build_coco_gt(
        images_dir=args.images,
        labels_dir=args.labels,
        names=args.names,
        out_json=out_dir / "gt_coco.json",
        area_mode=args.area_mode,
    )
    pred_bbox_json = convert_predictions_to_bbox(args.pred, out_dir / "pred_bbox_coco.json")
    metrics = evaluate_coco(gt_json, pred_bbox_json)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"GT json: {gt_json}")
    print(f"Prediction bbox json: {pred_bbox_json}")
    print(f"Metrics json: {out_dir / 'metrics.json'}")
