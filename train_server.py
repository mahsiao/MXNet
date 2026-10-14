"""Server full-training configuration.

Before starting, set the dataset root in Drone-obb.server.yaml for the server.
"""

import argparse
import time
import warnings

from ultralytics.models.dualyolo.model import DualYOLO

warnings.filterwarnings("ignore")


CONTRAST_MI_LAYERS = {
    "p3_ir_afab_sfem": [6, 12, 14],
}

MODEL_CONFIGS = {

    # "p3-offset": "mxrecode/alignment/yolo11-mid-p3-offset-alignment.yaml",
    "p3-no-offset": "mxrecode/alignment/yolo11-mid-p3-no-offset-alignment.yaml",
    "p2-offset": "mxrecode/alignment/yolo11-mid-p2-offset-alignment.yaml",
}

TRAIN_ARGS = dict(
    data="mxrecode/datasets/Drone-server.yaml",
    cache=False,
    imgsz=640,
    epochs=200,
    batch=64,
    close_mosaic=10,
    workers=8,
    device="0",
    optimizer="SGD",
    amp=True,
    # fraction=0.05,
    channels=4,
    project="runs/DVOBB",
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
