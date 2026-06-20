import warnings
warnings.filterwarnings('ignore')
import glob
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import DualYOLO
from ultralytics.engine.results import Results
from ultralytics.utils import ops

drone_ckpt = r'D:\CProject\MXNet\eval\ckpts\DV\DV-11n-mid.pt'
umod = r'E:\CProject\Dataset\AUMOD\train\visible\images\00007.jpg'
VIS_LINE_WIDTH = 2
DEFAULT_PROJECT = r"visualization\runs\obb"
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def image_to_label_path(image_path):
    path = Path(image_path)
    parts = list(path.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return path.parent.parent / "labels" / f"{path.stem}.txt"


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
                points = coords[:8].reshape(4, 2)
                labels.append((cls, points))
            elif coords.size == 5:
                # Fallback for xywhr labels is intentionally skipped here because DroneVehicle OBB labels are polygons.
                continue
    return labels


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
    return Results(image, path="", names=names or {}, obb=obb).plot(conf=False, line_width=line_width)


def save_gt_visualization(source, save_dir, names=None, line_width=VIS_LINE_WIDTH):
    source = Path(source)
    save_dir = Path(save_dir)
    label_path = image_to_label_path(source)
    labels = read_obb_labels(label_path)
    if not labels:
        print(f"No GT labels found: {label_path}")
        return

    rgb = cv2.imread(str(source))
    ir_path = Path(str(source).replace("visible", "infrared"))
    ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
    if rgb is None:
        print(f"RGB image not found: {source}")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_dir / f"{source.stem}_gt.png"), draw_gt(rgb, labels, names, line_width=line_width))

    if ir is not None:
        ir_bgr = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(save_dir / f"{source.stem}_gt_ir.png"), draw_gt(ir_bgr, labels, names, line_width=line_width))
    else:
        print(f"IR image not found: {ir_path}")


def collect_sources(source):
    """Return concrete image paths for GT visualization from a file, folder, glob, or list."""
    if isinstance(source, (list, tuple)):
        paths = []
        for item in source:
            paths.extend(collect_sources(item))
        return paths

    source_str = str(source)
    if any(ch in source_str for ch in "*?[]"):
        return [Path(p) for p in sorted(glob.glob(source_str, recursive=True)) if Path(p).suffix.lower() in IMAGE_SUFFIXES]

    path = Path(source_str)
    if path.is_dir():
        return [p for p in sorted(path.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    return []


def load_model(model):
    """Accept either an existing DualYOLO object or a checkpoint/config path."""
    return model if isinstance(model, DualYOLO) else DualYOLO(model)


def visualize_one(
    model,
    source,
    name="exp",
    project=DEFAULT_PROJECT,
    imgsz=640,
    line_width=VIS_LINE_WIDTH,
    save_gt=True,
    **predict_kwargs,
):
    """Run RGBT OBB prediction and optionally save matched RGB/IR GT visualizations.

    Args:
        model: A DualYOLO instance or a checkpoint/config path.
        source: Image path, folder, glob, or list of image paths.
        name: Experiment folder name under project.
        project: Output root directory.
        imgsz: Inference image size.
        line_width: Shared line width for prediction and GT plots.
        save_gt: Whether to draw GT files next to prediction files.
        **predict_kwargs: Extra arguments passed to model.predict, e.g. conf=0.25, device="0".
    """
    model = load_model(model)
    results = model.predict(
        source=source,
        imgsz=imgsz,
        project=project,
        name=name,
        show=False,
        save_frames=True,
        use_simotm="RGBT",
        channels=4,
        line_width=line_width,
        save=True,
        **predict_kwargs,
    )

    if save_gt:
        save_dir = getattr(results[0], "save_dir", None) if results else Path(project) / name
        names = results[0].names if results else None
        for image_path in collect_sources(source):
            save_gt_visualization(image_path, save_dir, names=names, line_width=line_width)
    return results


def visualize_jobs(jobs, **common_kwargs):
    """Run multiple visualization jobs.

    Example job:
        {
            "model": r"path/to/best.pt",
            "source": [r"path/to/00001.jpg", r"path/to/00002.jpg"],
            "name": "my_model_dv",
            "conf": 0.25,
        }
    """
    all_results = []
    for job in jobs:
        job = {**common_kwargs, **job}
        model = job.pop("model")
        source = job.pop("source")
        all_results.append(visualize_one(model=model, source=source, **job))
    return all_results


if __name__ == '__main__':
    jobs = [
        {
            "model": drone_ckpt,
            "source": r"E:\CProject\Dataset\DVOBB\train\visible\images\00007.jpg",
            "name": "dv_lcaf_p3_00007",
        },
        # More examples:
        # {
        #     "model": r"D:\CProject\MXNet\eval\ckpts\another_model.pt",
        #     "source": [
        #         r"E:\CProject\Dataset\DVOBB\val\visible\images\00001.jpg",
        #         r"E:\CProject\Dataset\DVOBB\val\visible\images\00002.jpg",
        #     ],
        #     "name": "another_model_dv_val",
        # },
        # {
        #     "model": drone_ckpt,
        #     "source": r"E:\CProject\Dataset\AUMOD\train\visible\images\*.jpg",
        #     "name": "aumod_lcaf_p3_samples",
        # },
    ]
    visualize_jobs(jobs, imgsz=640, line_width=VIS_LINE_WIDTH)
