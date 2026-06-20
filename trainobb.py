import warnings

from ultralytics.models.yolo.model import DualYOLO
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    model = DualYOLO('mxrecode/baseline/yolo11-mid-p3.yaml')
    # model.info(True,True)
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=r'mxrecode/datasets/DV128-obb.yaml',
                cache=False,
                imgsz=640,
                epochs=10,
                batch=8,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD',  # using SGD
                # lr0=0.002,
                # resume='', # last.pt path
                amp=True, # close amp
                # fraction=0.2,
                use_simotm="RGBT",
                channels=4,
                project='runs/UMOD',
                name='test-base',
                )
