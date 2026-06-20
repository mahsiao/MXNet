import argparse
import sys
import types
import warnings
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import DualYOLO
from visualization.feature_heatmap import (
    collect_sources,
    get_model_channels,
    get_pytorch_model,
    load_rgbt_tensor,
    overlay_heatmap,
    read_rgb_ir,
)
from visualization.gradcam_heatmap import normalize_heatmap

warnings.filterwarnings("ignore")

DEFAULT_PROJECT = Path("visualization/runs/attention")


def find_attention_modules(torch_model):
    return {
        name: module
        for name, module in torch_model.named_modules()
        if module.__class__.__name__ in {"Attention", "AAttn"}
    }


def patch_attention_module(module, store, name):
    original_forward = module.forward

    def forward_with_attention(self, x):
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        store[name] = attn.detach().float().mean(dim=(1, 2)).view(b, h, w).cpu()

        out = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        return self.proj(out)

    module.forward = types.MethodType(forward_with_attention, module)
    return original_forward


def restore_attention_modules(patched):
    for module, original_forward in patched:
        module.forward = original_forward


def save_attention_outputs(attn_map, source, save_dir, module_name, alpha):
    module_dir = Path(save_dir) / module_name.replace(".", "_")
    module_dir.mkdir(parents=True, exist_ok=True)

    heatmap = normalize_heatmap(attn_map)
    rgb, ir = read_rgb_ir(source)
    stem = Path(source).stem
    cv2.imwrite(str(module_dir / f"{stem}_raw_attention.png"), heatmap)
    cv2.imwrite(str(module_dir / f"{stem}_rgb_attention.png"), overlay_heatmap(rgb, heatmap, alpha))
    if ir is not None:
        cv2.imwrite(str(module_dir / f"{stem}_ir_attention.png"), overlay_heatmap(ir, heatmap, alpha))


def visualize_attention(
    model,
    source,
    modules=None,
    name="attention",
    project=DEFAULT_PROJECT,
    imgsz=640,
    alpha=0.45,
    device=None,
):
    sources = collect_sources(source)
    if not sources:
        raise FileNotFoundError(f"No image sources found from: {source}")

    model = model if isinstance(model, DualYOLO) else DualYOLO(model)
    if get_model_channels(model, default=4) != 4:
        raise ValueError("Attention script currently expects a 4-channel RGBT checkpoint.")

    torch_model = get_pytorch_model(model)
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    torch_model.to(device).eval()

    all_attention = find_attention_modules(torch_model)
    if not all_attention:
        raise RuntimeError("No Attention/AAttn modules found in this checkpoint.")

    selected = modules or list(all_attention)
    missing = [m for m in selected if m not in all_attention]
    if missing:
        raise KeyError(f"Attention modules not found: {missing}. Available: {list(all_attention)}")

    store = {}
    patched = [(all_attention[name], patch_attention_module(all_attention[name], store, name)) for name in selected]
    save_dir = Path(project) / name

    try:
        with torch.inference_mode():
            for source_path in sources:
                store.clear()
                image_tensor = load_rgbt_tensor(source_path, imgsz=imgsz, device=device)
                torch_model(image_tensor)
                for module_name in selected:
                    if module_name in store:
                        save_attention_outputs(store[module_name][0], source_path, save_dir, module_name, alpha)
                    else:
                        print(f"No attention captured for module: {module_name}")
    finally:
        restore_attention_modules(patched)

    return save_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Generate self-attention heatmaps for RGBT models.")
    parser.add_argument("--model", default=r"D:\CProject\MXNet\eval\ckpts\DV\DV-11n-mid.pt")
    parser.add_argument("--source", default=r"E:\CProject\Dataset\DVOBB\train\visible\images\00027.jpg")
    parser.add_argument("--modules", nargs="+", default=None)
    parser.add_argument("--name", default="dv_00027_attention")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = visualize_attention(
        model=args.model,
        source=args.source,
        modules=args.modules,
        name=args.name,
        project=args.project,
        imgsz=args.imgsz,
        alpha=args.alpha,
        device=args.device,
    )
    print(f"Attention heatmaps saved to: {out_dir}")
