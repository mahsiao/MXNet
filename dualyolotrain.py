import warnings
import argparse

from ultralytics.models.dualyolo.model import DualYOLO
warnings.filterwarnings('ignore')
"""
"baseline": "mxrecode/baseline/yolo11-mid-p3.yaml",

    # P3 single-direction cross attention
    "p3_ir_guided_rgb": "mxrecode/crossattention/yolo11-mid-p3-ir-guided-rgb.yaml",
    "p3_rgb_guided_ir": "mxrecode/crossattention/yolo11-mid-p3-rgb-guided-ir.yaml",

    # P3 bidirectional cross attention
    "p3_bidirectional_kv_guided": "mxrecode/crossattention/yolo11-mid-p3-bidirectional-kv-guided.yaml",
    "p3_bidirectional_value_guided": "mxrecode/crossattention/yolo11-mid-p3-bidirectional-value-guided.yaml",

    # P3/P4/P5 single-direction cross attention
    "mid_ir_guided_rgb": "mxrecode/crossattention/yolo11-mid-ir-guided-rgb.yaml",
    "mid_rgb_guided_ir": "mxrecode/crossattention/yolo11-mid-rgb-guided-ir.yaml",
    "mid_ir_value_guided_rgb": "mxrecode/crossattention/yolo11-mid-ir-value-guided-rgb.yaml",
    "mid_rgb_value_guided_ir": "mxrecode/crossattention/yolo11-mid-rgb-value-guided-ir.yaml",

    # P3/P4/P5 bidirectional cross attention
    "mid_bidirectional_kv_guided": "mxrecode/crossattention/yolo11-mid-bidirectional-kv-guided.yaml",
    "mid_bidirectional_value_guided": "mxrecode/crossattention/yolo11-mid-bidirectional-value-guided.yaml",
"""

MODEL_CONFIGS = {
    # P3 illumination/fusion comparison
    "p3_illumination_residual_fusion": "mxrecode/fusion/yolo11-mid-p3-illumination-residual-fusion.yaml",
    "p3_cifusion": "mxrecode/fusion/yolo11-mid-p3-cifusion.yaml",
    "p3_cifusion_v6": "mxrecode/fusion/yolo11-mid-p3-cifusion-v6.yaml",
    "p3_lifadd": "mxrecode/fusion/yolo11-mid-p3-lifadd.yaml",

    # P3/P4/P5 illumination/fusion comparison
    "mid_illumination_residual_fusion": "mxrecode/fusion/yolo11-mid-illumination-residual-fusion.yaml",
    "mid_cifusion": "mxrecode/fusion/yolo11-mid-cifusion.yaml",
    "mid_cifusion_v6": "mxrecode/fusion/yolo11-mid-cifusion-v6.yaml",
    "mid_lifadd": "mxrecode/fusion/yolo11-mid-lifadd.yaml",
}


TRAIN_ARGS = dict(
    data=r"mxrecode/datasets/DV128-obb.yaml",
    cache=False,
    imgsz=640,
    epochs=10,
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
)


def train_variant(variant, tag=None):
    model = DualYOLO(MODEL_CONFIGS[variant])
    # model.info(True, True)
    # model.load("yolov8n.pt")  # loading pretrain weights
    name = f"{tag}_{variant}" if tag else variant
    model.train(**TRAIN_ARGS, name=name)


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline and cross-modal enhancement variants.")
    parser.add_argument(
        "--variant",
        choices=[*MODEL_CONFIGS, "all"],
        default="baseline",
        help="Which model config to train. Use 'all' to run all variants sequentially.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional batch tag used as a prefix for experiment names, e.g. p3_ablation_v1.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    variants = list(MODEL_CONFIGS) if args.variant == "all" else [args.variant]
    for variant in variants:
        print(f"Training variant: {variant} ({MODEL_CONFIGS[variant]})")
        train_variant(variant, tag=args.tag)
