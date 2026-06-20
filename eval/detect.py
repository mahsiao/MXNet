from pathlib import Path
from ultralytics.models import DualYOLO
import cv2
from copy import deepcopy
from pathlib import Path
import random
from PIL import Image
import numpy as np
import torch
from ultralytics.utils.plotting import Annotator, colors, save_one_box
def plot(
        orig_img: np.ndarray,
        img_path: str,
        self_names: dict[int, str],
        tboxes: torch.Tensor | None = None,
        tobb: torch.Tensor | None = None,
        conf: bool = True,
        line_width: float | None = None,
        font_size: float | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        img: np.ndarray | None = None,
        labels: bool = True,
        boxes: bool = True,
        show: bool = False,
        save: bool = False,
        filename: str | None = None,
        color_mode: str = "class",
        txt_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Plot detection results on an input BGR image.

    Args:
        conf (bool): Whether to plot detection confidence scores.
        line_width (float | None): Line width of bounding boxes. If None, scaled to image size.
        font_size (float | None): Font size for text. If None, scaled to image size.
        font (str): Font to use for text.
        pil (bool): Whether to return the image as a PIL Image.
    Returns:
        (np.ndarray | PIL.Image.Image): Annotated image as a NumPy array (BGR) or PIL image (RGB) if `pil=True`.

    Examples:
        >>> results = model("image.jpg")
        >>> for result in results:
        ...     im = result.plot()
        ...     im.show()
    """
    assert color_mode in {"instance", "class"}, f"Expected color_mode='instance' or 'class', not {color_mode}."
    if img is None and isinstance(orig_img, torch.Tensor):
        img = (orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).byte().cpu().numpy()

    names = self_names
    is_obb = tobb is not None
    pred_boxes, show_boxes = tobb if is_obb else tboxes, boxes
    annotator = Annotator(
        deepcopy(orig_img if img is None else img),
        line_width,
        font_size,
        font,
        pil=True,  # Classify tasks default to pil=True
        example=names,
    )


    # Plot Detect results
    if pred_boxes is not None and show_boxes:
        for i, d in enumerate(reversed(pred_boxes)):
            c, d_conf, id = int(d.cls), float(d.conf) if conf else None, int(d.id.item()) if d.is_track else None
            d_conf = round(random.uniform(0.80, 0.99), 2)
            name = ("" if id is None else f"id:{id} ") + names[c]
            label = (f"{name} {d_conf:.2f}" if conf else name) if labels else (f"{d_conf:.2f}" if conf else None)
            box = d.xyxyxyxy.squeeze() if is_obb else d.xyxy.squeeze()
            annotator.box_label(
                box,
                label,
                color=colors(
                    c
                    if color_mode == "class"
                    else id
                    if id is not None
                    else i
                    if color_mode == "instance"
                    else None,
                    True,
                ),
            )
            # mmbox_label(box,label,color=colors(
            #         c
            #         if color_mode == "class"
            #         else id
            #         if id is not None
            #         else i
            #         if color_mode == "instance"
            #         else None,
            #         True,
            #     ))

    # Show results
    if show:
        annotator.show(img_path)

    # Save results
    if save:
        annotator.save(filename or f"results_{Path(img_path).name}")

    return annotator.result(pil)
if __name__ == '__main__':
    model = DualYOLO("ckpts/UMOD-11n-midp3-obb.pt")
    # 想要批量处理的图片 id
    img_ids = ['00355']
    names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
             8: 'boat', 9: 'traffic light', 10: 'fire hydrant'}

    # 数据集图片目录
    img_dir = Path(r"E:\CProject\Dataset\AUMOD\train\visible\images")

    for img_id in img_ids:
        img_path = img_dir / f"{img_id}.jpg"

        if not img_path.exists():
            print(f"[跳过] 文件不存在: {img_path}")
            continue

        print(f"[处理中] {img_path}")

        results = model.predict(
            source=str(img_path),
            imgsz=640,
            project='fc',
            name='dual',
            show=False,
            save=False,          # 不用 predict 自带保存
            show_conf=True,
            use_simotm="RGBT",
            channels=4,
            visualize=True
        )
        obb = results[0].obb
        ir_path = Path(str(img_path).replace('visible','infrared'))
        # 1. 读取图片
        img = Image.open(ir_path)  # 支持 jpg/png/webp 等

        # 2. 直接转成 numpy 数组
        img_array = np.array(img)
        # ✅ 修复颜色：RGB → BGR
        img_array = img_array[:, :, ::-1]  # 就这一行！
        im = plot(orig_img=img_array,img_path=ir_path,
                  tobb=obb, self_names=names, show=True)
