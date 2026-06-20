import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from prettytable import PrettyTable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.models import DualYOLO
from ultralytics.utils.torch_utils import model_info

warnings.filterwarnings("ignore")

# python batch_val.py ckpts/DV
DEFAULT_DATA = ROOT / "eval" / "datasets" / "DVOBB.yaml"
DEFAULT_PROJECT = ROOT / "eval" / "runs" / "batch_val"


def get_weight_size(path):
    return f"{os.stat(path).st_size / 1024 / 1024:.1f}"


def collect_weights(items):
    weights = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            weights.extend(sorted(path.rglob("*.pt")))
        elif path.is_file() and path.suffix.lower() == ".pt":
            weights.append(path)
        else:
            print(f"Skip invalid weight path: {item}")
    return weights


def safe_metric(results_dict, key, default=np.nan):
    return results_dict.get(key, default)


def format_duration(seconds):
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def summarize_result(model, result, weight_path):
    preprocess = result.speed["preprocess"]
    inference = result.speed["inference"]
    postprocess = result.speed["postprocess"]
    total = preprocess + inference + postprocess
    n_l, n_p, n_g, flops = model_info(model.model)

    length = result.box.p.size
    f1 = float(np.mean(result.box.f1[:length])) if length else np.nan
    map75 = float(np.mean(result.box.all_ap[:length, 5])) if length else np.nan

    return {
        "model": Path(weight_path).stem,
        "weight": str(weight_path),
        "save_dir": str(result.save_dir),
        "GFLOPs": f"{flops:.1f}",
        "params": n_p,
        "size_MB": get_weight_size(weight_path),
        "precision": safe_metric(result.results_dict, "metrics/precision(B)"),
        "recall": safe_metric(result.results_dict, "metrics/recall(B)"),
        "F1": f1,
        "mAP50": safe_metric(result.results_dict, "metrics/mAP50(B)"),
        "mAP75": map75,
        "mAP50-95": safe_metric(result.results_dict, "metrics/mAP50-95(B)"),
        "preprocess_ms": preprocess,
        "inference_ms": inference,
        "postprocess_ms": postprocess,
        "FPS_total": 1000 / total if total else np.nan,
        "FPS_inference": 1000 / inference if inference else np.nan,
    }


def write_summary(rows, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "summary.csv"
    txt_path = save_dir / "summary.txt"

    if not rows:
        return csv_path, txt_path

    fieldnames = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    table = PrettyTable()
    table.title = "Batch Validation Summary"
    table.field_names = ["model", "P", "R", "F1", "mAP50", "mAP75", "mAP50-95", "FPS", "GFLOPs", "params"]
    for row in rows:
        table.add_row(
            [
                row["model"],
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['F1']:.4f}",
                f"{row['mAP50']:.4f}",
                f"{row['mAP75']:.4f}",
                f"{row['mAP50-95']:.4f}",
                f"{row['FPS_total']:.2f}",
                row["GFLOPs"],
                f"{row['params']:,}",
            ]
        )

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(str(table))
        f.write("\n")

    print(table)
    return csv_path, txt_path


def validate_one(weight_path, args):
    start = time.perf_counter()
    model = DualYOLO(str(weight_path))
    result = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        use_simotm="RGBT",
        channels=4,
        save_json=args.save_json,
        project=str(args.project),
        name=Path(weight_path).stem,
        device=args.device,
    )
    row = summarize_result(model, result, weight_path)
    row["val_seconds"] = time.perf_counter() - start
    row["val_time"] = format_duration(row["val_seconds"])
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Batch validate DualYOLO OBB models on one dataset.")
    parser.add_argument(
        "weights",
        nargs="+",
        help="One or more .pt files or directories containing .pt weights.",
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset YAML.")
    parser.add_argument("--split", default="val", help="Dataset split.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--save-json", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.data = Path(args.data)
    args.project = Path(args.project)
    total_start = time.perf_counter()

    weights = collect_weights(args.weights)
    if not weights:
        raise FileNotFoundError("No .pt weights found.")

    rows = []
    for i, weight_path in enumerate(weights, start=1):
        print(f"[{i}/{len(weights)}] Validating: {weight_path}")
        row = validate_one(weight_path, args)
        rows.append(row)
        print(f"[{i}/{len(weights)}] Finished: {row['model']}, duration: {row['val_time']}")

    csv_path, txt_path = write_summary(rows, args.project)
    print(f"Summary saved to: {csv_path}")
    print(f"Summary table saved to: {txt_path}")

    total_duration = time.perf_counter() - total_start
    print("\nValidation time summary:")
    for row in rows:
        print(f"  {row['model']}: {row['val_time']}")
    print(f"  total: {format_duration(total_duration)}")
