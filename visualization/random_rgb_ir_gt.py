import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.engine.results import Results
from ultralytics.utils import ops

VIS_LINE_WIDTH = 2
DEFAULT_VISIBLE_IMAGES = r"E:\CProject\Dataset\DVOBB\train\visible\images"
DEFAULT_VISIBLE_LABELS = r"E:\CProject\Dataset\DVOBB\train\visible\fixlabel"
DEFAULT_SAVE_DIR = r"visualization\runs\gt_random_rgb_ir"
DEFAULT_NAMES = ["car", "bus", "van", "truck", "freight_car"]
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def image_to_label_path(image_path):
    path = Path(image_path)
    parts = list(path.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return path.parent.parent / "labels" / f"{path.stem}.txt"


def label_to_image_path(label_path):
    label_path = Path(label_path)
    image_dir = label_path.parent.parent / "images"
    for suffix in sorted(IMAGE_SUFFIXES):
        image_path = image_dir / f"{label_path.stem}{suffix}"
        if image_path.exists():
            return image_path
    return image_dir / f"{label_path.stem}.jpg"


def visible_to_ir_label_path(visible_label_path):
    parts = list(Path(visible_label_path).parts)
    for i, part in enumerate(parts):
        if part == "visible":
            parts[i] = "infrared"
            return Path(*parts)
    return Path(str(visible_label_path).replace("visible", "infrared"))


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


def names_to_dict(names):
    if names is None:
        names = DEFAULT_NAMES
    if isinstance(names, dict):
        return names
    return {i: name for i, name in enumerate(names)}


def draw_gt(image, labels, names=None, line_width=VIS_LINE_WIDTH):
    h, w = image.shape[:2]
    classes, boxes = [], []
    for cls, points in labels:
        pts = points.copy()
        if pts.max() <= 1.01:
            pts[:, 0] *= w
            pts[:, 1] *= h
        classes.append(cls)
        boxes.append(pts.reshape(-1))

    if not boxes:
        return image

    boxes = torch.from_numpy(np.stack(boxes).astype(np.float32))
    xywhr = ops.xyxyxyxy2xywhr(boxes)
    conf = torch.ones((len(boxes), 1), dtype=xywhr.dtype)
    cls = torch.tensor(classes, dtype=xywhr.dtype).view(-1, 1)
    obb = torch.cat((xywhr, conf, cls), dim=1)
    return Results(image, path="", names=names_to_dict(names), obb=obb).plot(conf=False, line_width=line_width)


def visible_to_ir_path(visible_path):
    parts = list(Path(visible_path).parts)
    for i, part in enumerate(parts):
        if part == "visible":
            parts[i] = "infrared"
            return Path(*parts)
    return Path(str(visible_path).replace("visible", "infrared"))


def has_labels(image_path):
    label_path = image_to_label_path(image_path)
    return label_path.exists() and bool(read_obb_labels(label_path))


def collect_images(image_dir, require_label=True):
    image_dir = Path(image_dir)
    images = [p for p in sorted(image_dir.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
    if require_label:
        images = [
            p
            for p in images
            if has_labels(p) and visible_to_ir_path(p).exists() and has_labels(visible_to_ir_path(p))
        ]
    return images


def collect_visible_labels(label_dir, require_ir_label=True):
    label_dir = Path(label_dir)
    labels = [p for p in sorted(label_dir.rglob("*.txt")) if read_obb_labels(p)]
    if require_ir_label:
        labels = [p for p in labels if visible_to_ir_label_path(p).exists() and read_obb_labels(visible_to_ir_label_path(p))]
    return labels


def resize_to_height(image, height):
    h, w = image.shape[:2]
    if h == height:
        return image
    width = max(1, round(w * height / h))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_pair_canvas(rgb_gt, ir_gt):
    target_h = max(rgb_gt.shape[0], ir_gt.shape[0])
    rgb_gt = resize_to_height(rgb_gt, target_h)
    ir_gt = resize_to_height(ir_gt, target_h)
    return np.concatenate([rgb_gt, ir_gt], axis=1)


def make_grid(rows, max_width=1800):
    resized_rows = []
    for row in rows:
        h, w = row.shape[:2]
        if w > max_width:
            row = cv2.resize(row, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)
        resized_rows.append(row)

    width = max(row.shape[1] for row in resized_rows)
    padded_rows = []
    for row in resized_rows:
        pad = width - row.shape[1]
        if pad:
            row = cv2.copyMakeBorder(row, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        padded_rows.append(row)
    return np.concatenate(padded_rows, axis=0)


def visualize_random_gt(
    visible_images=DEFAULT_VISIBLE_IMAGES,
    visible_labels=None,
    save_dir=DEFAULT_SAVE_DIR,
    num=10,
    seed=None,
    line_width=VIS_LINE_WIDTH,
    names=None,
    show=False,
    require_label=True,
):
    if visible_labels:
        sources = collect_visible_labels(visible_labels, require_ir_label=require_label)
        if not sources:
            raise FileNotFoundError(f"No labels found in {visible_labels}")
    else:
        sources = collect_images(visible_images, require_label=require_label)
        if not sources:
            raise FileNotFoundError(f"No images found in {visible_images}")

    rng = random.Random(seed)
    samples = rng.sample(sources, k=min(num, len(sources)))
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for source_path in samples:
        if visible_labels:
            rgb_label_path = source_path
            visible_path = label_to_image_path(rgb_label_path)
            ir_label_path = visible_to_ir_label_path(rgb_label_path)
            ir_path = label_to_image_path(ir_label_path)
        else:
            visible_path = source_path
            rgb_label_path = image_to_label_path(visible_path)
            ir_path = visible_to_ir_path(visible_path)
            ir_label_path = image_to_label_path(ir_path)

        rgb_labels = read_obb_labels(rgb_label_path)
        rgb = cv2.imread(str(visible_path))
        ir_labels = read_obb_labels(ir_label_path)
        ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)

        if rgb is None:
            print(f"Skip missing RGB image: {visible_path}")
            continue
        if ir is None:
            print(f"Skip missing IR image: {ir_path}")
            continue

        ir_bgr = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        rgb_gt = draw_gt(rgb, rgb_labels, names=names, line_width=line_width)
        ir_gt = draw_gt(ir_bgr, ir_labels, names=names, line_width=line_width)

        cv2.imwrite(str(save_dir / f"{visible_path.stem}_rgb_gt.jpg"), rgb_gt)
        cv2.imwrite(str(save_dir / f"{visible_path.stem}_ir_gt.jpg"), ir_gt)
        rows.append(make_pair_canvas(rgb_gt, ir_gt))
        print(
            f"{visible_path.name}: "
            f"RGB={visible_path} label={rgb_label_path} labels={len(rgb_labels)} | "
            f"IR={ir_path} label={ir_label_path} labels={len(ir_labels)}"
        )

    if rows:
        grid = make_grid(rows)
        grid_path = save_dir / f"random_{len(rows)}_rgb_ir_gt.jpg"
        cv2.imwrite(str(grid_path), grid)
        print(f"Saved grid: {grid_path}")

        if show:
            cv2.imshow("RGB GT | IR GT", grid)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Randomly visualize RGB/IR OBB GT pairs.")
    parser.add_argument("--visible-images", default=DEFAULT_VISIBLE_IMAGES, help="Path to visible/images.")
    parser.add_argument(
        "--visible-labels",
        default=DEFAULT_VISIBLE_LABELS,
        help="Path to visible labels/fixlabel. When set, samples labels first and finds images by stem.",
    )
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help="Directory for visualized outputs.")
    parser.add_argument("--num", type=int, default=10, help="Number of random image pairs.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling.")
    parser.add_argument("--line-width", type=int, default=VIS_LINE_WIDTH, help="OBB line width.")
    parser.add_argument("--names", nargs="+", default=DEFAULT_NAMES, help="Class names in label-id order.")
    parser.add_argument("--show", action="store_true", help="Open a cv2 window after saving the grid.")
    parser.add_argument("--allow-empty-labels", action="store_true", help="Allow images with no GT labels.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_random_gt(
        visible_images=args.visible_images,
        visible_labels=args.visible_labels,
        save_dir=args.save_dir,
        num=args.num,
        seed=args.seed,
        line_width=args.line_width,
        names=args.names,
        show=args.show,
        require_label=not args.allow_empty_labels,
    )
