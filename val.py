from ultralytics import YOLO

# model = YOLOv10.from_pretrained('jameslahm/yolov10n')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
model = YOLO('runs/train/BS-YOLO/last.pt')

model.val(data='ultralytics/cfg/datasets/car.yaml', batch=64)