import os
from ultralytics import YOLO

# Paths
yaml_path = "yolov8.yaml"
model_size = "yolov8n.pt"
project_name = "training_yolov8_luna16"
experiment_name = "yolov8n_lung_nodules"

# Load model
model = YOLO(model_size)

# Custom augmentations suitable for CT (disable color and mosaic, limit flip/scale)
model.train(
    data=yaml_path,
    epochs=300,
    imgsz=512,
    batch=16,
    patience=200,
    project=project_name,
    name=experiment_name,
    exist_ok=True,
    device=0,
    workers=4,
    pretrained=True,
    val=True,
    hsv_h=0.0,        # disable hue
    hsv_s=0.0,        # disable saturation
    hsv_v=0.0,        # disable brightness
    degrees=5.0,      # limit rotation
    scale=0.1,        # limit zooming
    shear=0.0,        # disable shearing
    perspective=0.0,  # disable perspective
    flipud=0.0,       # disable vertical flip (not valid for CTs)
    fliplr=0.3,       # allow horizontal flip (optional)
    mosaic=0.0,       # disable mosaic
    mixup=0.0,        # disable mixup
    copy_paste=0.0    # disable copy-paste
)
