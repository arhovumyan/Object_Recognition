#!/bin/bash

# Download YOLOv8s model if not present
if [ ! -f "yolov8s.pt" ]; then
    echo "Downloading YOLOv8s model..."
    python3 -c "
import ultralytics
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
print('YOLOv8s model downloaded successfully!')
"
else
    echo "YOLOv8s model already exists."
fi

echo "Model download complete!"
