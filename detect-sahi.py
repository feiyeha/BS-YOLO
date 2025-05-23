from sahi.predict import predict

predict(
    model_type="ultralytics",
    model_path="runs/train/exp9/weights/best.pt",
    model_device="cuda:0",  # or 'cuda:0'
    model_confidence_threshold=0.4,
    source="../yolov5-master/detect/",
    slice_height=800,
    slice_width=800,
    overlap_height_ratio=0,
    overlap_width_ratio=0,
)