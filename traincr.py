import warnings
import argparse
import time

from ultralytics.models.dualyolo.model import DualYOLO
warnings.filterwarnings('ignore')


MODEL_CONFIGS = {
    # P3 illumination/fusion comparison
    # "p3_illumination_residual_fusion": "mxrecode/fusion/yolo11-mid-p3-illumination-residual-fusion.yaml",
    # "p3_cifusion": "mxrecode/fusion/yolo11-mid-p3-cifusion.yaml",
    # "p3_cifusion_v6": "mxrecode/fusion/yolo11-mid-p3-cifusion-v6.yaml",
    # "p3_lifadd": "mxrecode/fusion/yolo11-mid-p3-lifadd.yaml",
    "p3_mid-p3":"mxrecode/baseline/yolo11-mid-p3.yaml",

    # P3/P4/P5 illumination/fusion comparison
    # "mid_illumination_residual_fusion": "mxrecode/fusion/yolo11-mid-illumination-residual-fusion.yaml",
    # "mid_cifusion": "mxrecode/fusion/yolo11-mid-cifusion.yaml",
    # "mid_cifusion_v6": "mxrecode/fusion/yolo11-mid-cifusion-v6.yaml",
    # "mid_lifadd": "mxrecode/fusion/yolo11-mid-lifadd.yaml",
}

CONTRAST_MI_LAYERS = {
    "p3_illumination_residual_fusion": [7, 13, 15],
    "p3_cifusion": [6, 12, 15],
    "p3_mid-p3": [6, 12, 14],
    "p3_cifusion_v6": [6, 12, 15],
    "p3_lifadd": [7, 13, 15],
    "mid_illumination_residual_fusion": [7, 17, 22],
    "mid_cifusion": [6, 16, 23],
    "mid_cifusion_v6": [6, 16, 23],
    "mid_lifadd": [7, 17, 22],
}

TRAIN_ARGS = dict(
    data=r"mxrecode/datasets/DV128-obb.yaml",
    cache=False,
    imgsz=640,
    epochs=100,
    batch=16,
    close_mosaic=5,
    workers=0,
    device="0",
    optimizer="SGD",
    # lr0=0.002,
    # resume="",  # last.pt path
    amp=True,
    # fraction=0.2,
    channels=4,
    project="DVOBB",
    contrast_mi_gain=0.1,
    contrast_mi_stop_ratio=0.3,
    contrast_mi_stop_epoch=-1,
)


def train_variant(variant, tag=None):
    model = DualYOLO(MODEL_CONFIGS[variant])
    # model.info(True, True)
    # model.load("yolov8n.pt")  # loading pretrain weights
    name = f"{tag}_{variant}" if tag else variant
    train_args = {**TRAIN_ARGS, "contrast_mi_layers": CONTRAST_MI_LAYERS[variant]}
    start = time.perf_counter()
    model.train(**train_args, name=name)
    return time.perf_counter() - start


def format_duration(seconds):
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline and P3 cross-modal enhancement variants.")
    parser.add_argument(
        "--variant",
        choices=[*MODEL_CONFIGS, "all"],
        default="all",
        help="Which model config to train. Use 'all' to run all variants sequentially.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional batch tag used as a prefix for experiment names, e.g. p3_ablation_v1.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline when --variant all is used.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    total_start = time.perf_counter()
    durations = []
    variants = list(MODEL_CONFIGS) if args.variant == "all" else [args.variant]
    if args.skip_baseline:
        variants = [v for v in variants if v != "baseline"]
    for variant in variants:
        print(f"Training variant: {variant} ({MODEL_CONFIGS[variant]})")
        duration = train_variant(variant, tag=args.tag)
        durations.append((variant, duration))
        print(f"Finished variant: {variant}, duration: {format_duration(duration)}")

    total_duration = time.perf_counter() - total_start
    print("\nTraining time summary:")
    for variant, duration in durations:
        print(f"  {variant}: {format_duration(duration)}")
    print(f"  total: {format_duration(total_duration)}")
