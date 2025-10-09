#!/usr/bin/env python3

"""
CPU-Only Object Detection Script
Forces CPU usage to avoid CUDA compatibility issues

Usage:
    python3 src/live_object_detection_cpu.py
"""

import os
# Force CPU usage by hiding CUDA devices
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch

def main():
    print("🚀 CPU-Only Object Detection")
    print("=" * 40)
    
    # Verify CPU mode
    print(f"PyTorch device: CPU (forced)")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Load YOLO model (will use CPU automatically)
    print("Loading YOLOv8n model...")
    
    # Look for model in models directory
    model_paths = [
        'models/yolov8n.pt',
        '../models/yolov8n.pt',
        'yolov8n.pt'
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("❌ Could not find YOLO model file")
        print("Available files:")
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.pt'):
                    print(f"  {os.path.join(root, file)}")
        return
    
    print(f"Using model: {model_path}")
    model = YOLO(model_path)
    
    # Open camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        print("Trying alternative camera devices...")
        for i in range(1, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"✅ Successfully opened camera device {i}")
                break
        else:
            print("❌ Failed to open any camera device")
            return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("✅ Camera opened successfully")
    print("Press 'q' to quit")
    print("=" * 40)
    
    # Target objects to highlight
    target_objects = ['person', 'cell phone', 'laptop', 'mouse', 'tv', 
                      'bottle', 'cup', 'book', 'keyboard', 'chair']
    
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    # Frame skipping for CPU performance
    frame_skip = 2  # Process every 2nd frame
    frame_count = 0
    last_detections = []
    last_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        frame_count += 1
        
        # Only run detection on every Nth frame
        if frame_count % frame_skip == 0:
            try:
                # Run detection (CPU-only)
                results = model(frame, 
                              conf=0.3, 
                              verbose=False, 
                              device='cpu',
                              imgsz=320)  # Smaller for CPU
                
                last_detections = results
                last_frame = frame.copy()
            except Exception as e:
                print(f"❌ Detection error: {e}")
                continue
        else:
            # Reuse last detection results
            results = last_detections
            frame = last_frame if last_frame is not None else frame
        
        # Draw results
        if results:
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        
                        # Choose color based on target objects
                        if class_name in target_objects:
                            color = (0, 255, 0)  # Green
                            thickness = 2
                        else:
                            color = (255, 0, 0)  # Blue
                            thickness = 1
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Draw label
                        label = f"{class_name[:8]} {confidence:.1f}"
                        cv2.putText(frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Calculate and display FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        fps_text = f"FPS: {fps} (CPU)"
        cv2.putText(frame, fps_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display frame
        cv2.imshow('CPU-Only Object Detection', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done!")

if __name__ == "__main__":
    main()
