"""Local smoke-test training configuration.

This file is intentionally lightweight. Use trainobb_server.py for a full run.
"""

import argparse
import time
import warnings

from ultralytics.models.dualyolo.model import DualYOLO

warnings.filterwarnings("ignore")


MODEL_CONFIGS = {
    "p3_ir_afab_sfem": "mxrecode/baseline/yolo11-mid-p3-ir-afab-sfem.yaml",
}

TRAIN_ARGS = dict(
    data="mxrecode/datasets/DV128-obb.yaml",
    cache=False,
    imgsz=640,
    epochs=1,
    batch=2,
    close_mosaic=0,
    workers=0,
    device="0",
    optimizer="SGD",
    amp=True,
    fraction=0.05,
    channels=4,
    project="runs/local_test/DVOBB",
)

CONTRAST_MI_LAYERS = {
    "p3_ir_afab_sfem": [6, 12, 14],
}
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
    parser = argparse.ArgumentParser(description="Run a lightweight local OBB training smoke test.")
    parser.add_argument("--variant", choices=[*MODEL_CONFIGS, "all"], default="all")
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    variants = list(MODEL_CONFIGS) if args.variant == "all" else [args.variant]
    for variant in variants:
        print(f"Local test variant: {variant} ({MODEL_CONFIGS[variant]})")
        duration = train_variant(variant, tag=args.tag)
        print(f"Finished {variant} in {duration:.1f}s")
