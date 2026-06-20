import warnings
warnings.filterwarnings('ignore')

import numpy as np
from prettytable import PrettyTable
from ultralytics.models import DualYOLO
# Overexposure : 223
# Normal       : 262
# Twilight     : PreTwilight + PostTwilight = 146
# Dim          : 281
# Night        : NearNight + Night = 591

if __name__ == '__main__':

    # 训练好的权重路径
    model_path = 'ckpts/UMOD-11-lacf-concat.pt'

    # 加载模型，只加载一次
    model = DualYOLO(model_path)

    # 3 个光照子集 yaml
    light_yaml_dict = {
        "Merged_Normal": r"datasets/UMOD_light/merged_normal.yaml",
        "Merged_Dim": r"datasets/UMOD_light/merged_dim.yaml",
        "Night": r"datasets/UMOD_light/night.yaml",
    }

    # 保存每个光照条件下的 mAP50
    map50_results = {}

    for light_name, yaml_path in light_yaml_dict.items():
        print("\n" + "=" * 80)
        print(f"正在验证光照子集：{light_name}")
        print(f"使用 yaml：{yaml_path}")
        print("=" * 80)

        result = model.val(
            data=yaml_path,
            split='val',
            imgsz=640,
            batch=16,
            use_simotm="RGBT",
            channels=4,
            project='UMOD_light_val',
            name=f'yolo11-mid-p3-obb-{light_name}',
            plots=False,
            save_json=False,
            verbose=False,
        )

        # 取 mAP50
        # 你的原代码里 OBB 任务对应的是 result.results_dict['metrics/mAP50(B)']
        map50 = result.results_dict['metrics/mAP50(B)']

        map50_results[light_name] = map50

        print(f"{light_name} mAP50: {map50:.4f}")

    # 输出总表
    print("\n" + "-" * 30 + " 不同光照条件 mAP50 结果 " + "-" * 30)

    table = PrettyTable()
    table.field_names = ["Light Condition", "mAP50"]

    for light_name, map50 in map50_results.items():
        table.add_row([light_name, f"{map50:.4f}"])

    avg_map50 = np.mean(list(map50_results.values()))
    table.add_row(["Average", f"{avg_map50:.4f}"])

    print(table)