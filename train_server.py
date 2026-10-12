"""Server full-training configuration.

Before starting, set the dataset root in Drone-obb.server.yaml for the server.
"""

import argparse
import time
import warnings

from ultralytics.models.dualyolo.model import DualYOLO

warnings.filterwarnings("ignore")


MODEL_CONFIGS = {
    "p3_ir_afab_sfem": "mxrecode/baseline/yolo11-mid-p3-ir-afab-sfem.yaml",
}

CONTRAST_MI_LAYERS = {
    "p3_ir_afab_sfem": [6, 12, 14],
}

TRAIN_ARGS = dict(
    data="mxrecode/datasets/Drone-obb.server.yaml",
    cache=False,
    imgsz=640,
    epochs=100,
    batch=16,
    close_mosaic=5,
    workers=8,
    device="0",
    optimizer="SGD",
    amp=True,
    channels=4,
    project="runs/server/DVOBB",
    contrast_mi_gain=0.1,
    contrast_mi_stop_ratio=0.3,
    contrast_mi_stop_epoch=-1,
)


def train_variant(variant, tag=None):
    model = DualYOLO(MODEL_CONFIGS[variant])
    name = f"{tag}_{variant}" if tag else variant
    train_args = {**TRAIN_ARGS}
    if variant in CONTRAST_MI_LAYERS:
        train_args["contrast_mi_layers"] = CONTRAST_MI_LAYERS[variant]
    start = time.perf_counter()
    model.train(**train_args, name=name)
    return time.perf_counter() - start


def parse_args():
    parser = argparse.ArgumentParser(description="Run full OBB training on the server.")
    parser.add_argument("--variant", choices=[*MODEL_CONFIGS, "all"], default="all")
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    total_start = time.perf_counter()
    variants = list(MODEL_CONFIGS) if args.variant == "all" else [args.variant]
    for variant in variants:
        print(f"Server training variant: {variant} ({MODEL_CONFIGS[variant]})")
        duration = train_variant(variant, tag=args.tag)
        print(f"Finished {variant} in {duration:.1f}s")
    print(f"Total training time: {time.perf_counter() - total_start:.1f}s")
