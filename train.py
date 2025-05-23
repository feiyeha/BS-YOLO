import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
     # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLO(model=r'ultralytics/cfg/models/11/yolo11old.yaml')
    # model.load('yolo11n.pt')
    # v9/yolov9t.yaml
    model.train(data=r'car.yaml',
                imgsz=640,
                epochs=200,
                batch=-1,
                workers=8,
                device='',
                optimizer='SGD',
                close_mosaic=20,
                resume=False,
                project='runs/train',
                name='1',
                single_cls=False,
                cache=False,
                )