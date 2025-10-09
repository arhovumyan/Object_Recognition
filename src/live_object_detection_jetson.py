#!/usr/bin/env python3

"""
Live Object Detection for Jetson Nano (OPTIMIZED)
Uses webcam for real-time object detection with Jetson-specific optimizations.

Jetson Optimizations:
- Lower resolution for better performance
- YOLOv8n model (lighter than YOLOv8s)
- Frame skipping for CPU efficiency
- CUDA acceleration when available
- Jetson-specific camera settings

Usage:
    python3 src/live_object_detection_jetson.py
    
Press 'q' to quit
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch


def main():
    """Run live object detection on Jetson Nano."""
    print("=" * 60)
    print("JETSON NANO - Live Object Detection with YOLOv8n")
    print("=" * 60)
    
    # Check if CUDA is available (should be on Jetson)
    use_gpu = torch.cuda.is_available()
    
    if use_gpu:
        print(f"Jetson GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        device = 'cuda'
        print("Using Jetson GPU acceleration")
    else:
        print("No GPU detected - running on CPU")
        device = 'cpu'
    
    print(f"\nUsing device: {device.upper()}")
    
    # Load YOLO model - use YOLOv8n for Jetson (lighter model)
    model_name = 'yolov8n.pt'

    # Look for model in models directory first
    candidate_paths = [
        os.path.join('..', 'models', model_name),  # ../models/yolov8n.pt
        os.path.join('models', model_name),        # models/yolov8n.pt
        os.path.join('..', model_name),            # ../yolov8n.pt
        model_name                                 # fallback
    ]

    model_file = None
    for p in candidate_paths:
        if os.path.exists(p):
            model_file = p
            break

    if model_file is None:
        model_file = model_name

    print(f"Loading {model_file} model...")
    model = YOLO(model_file)
    
    if use_gpu:
        try:
            model.to(device)
            print("Model moved to Jetson GPU")
        except Exception as e:
            print(f"Error moving model to GPU: {e}")
            print("Falling back to CPU...")
            device = 'cpu'
            use_gpu = False
    
    print("Model loaded successfully!")
    
    # Open webcam - try different camera indices for Jetson
    print("Opening camera...")
    camera_indices = [0, 1, 2]  # Common camera indices
    cap = None
    
    for i in camera_indices:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Successfully opened camera device {i}")
            break
    
    if cap is None or not cap.isOpened():
        print("Error: Could not open any camera")
        print("Available video devices:")
        os.system("ls /dev/video* 2>/dev/null || echo 'No video devices found'")
        return
    
    # Set camera properties optimized for Jetson
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution for Jetson
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Minimize latency
    cap.set(cv2.CAP_PROP_FPS, 15)            # Lower FPS for stability
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # Get actual camera resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")
    print()
    print("=" * 60)
    print("Detection running! Press 'q' to quit")
    print("=" * 60)
    
    # Target objects to highlight
    target_objects = ['person', 'cell phone', 'laptop', 'mouse', 'tv', 
                      'bottle', 'cup', 'book', 'keyboard', 'chair']
    
    # FPS calculation
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    # Frame skipping for Jetson performance
    frame_skip = 2 if not use_gpu else 1  # Skip more frames on CPU
    frame_count = 0
    
    # Store last detection results for skipped frames
    last_detections = []
    last_frame = None
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab frame")
            break
        
        frame_count += 1
        
        # Only run detection on every Nth frame
        if frame_count % frame_skip == 0:
            # Run YOLO detection with Jetson-optimized settings
            inference_size = 320  # Smaller for Jetson
            max_detections = 20   # Fewer detections for performance
            
            try:
                results = model(
                    frame, 
                    conf=0.4,           # Slightly higher confidence for Jetson
                    iou=0.5,            # IOU threshold for NMS
                    max_det=max_detections,
                    verbose=False,
                    imgsz=inference_size,
                    half=False,         # Disable half precision for stability
                    device=device
                )
            except Exception as e:
                print(f"Detection error: {e}")
                continue
            
            # Store results for next frames
            last_detections = results
            last_frame = frame.copy()
        else:
            # Reuse last detection results
            results = last_detections
            frame = last_frame if last_frame is not None else frame
        
        # Process results
        for result in results:
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Get detection info
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                    
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu())
                    class_id = int(box.cls[0].cpu())
                    class_name = model.names[class_id]
                    
                    # Choose color based on whether it's a target object
                    if class_name in target_objects:
                        color = (0, 255, 0)  # Green for target objects
                        thickness = 2
                    else:
                        color = (255, 0, 0)  # Blue for other objects
                        thickness = 1
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Prepare label
                    label = f"{class_name[:8]} {confidence:.1f}"  # Shorter labels
                    
                    # Draw label
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,  # Smaller font for Jetson
                        color,
                        1,
                        cv2.LINE_AA
                    )
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Display FPS with device info
        device_display = "Jetson GPU" if use_gpu else "Jetson CPU"
        fps_text = f"FPS: {fps} | {device_display}"
        cv2.putText(
            frame,
            fps_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )
        
        # Display frame
        cv2.imshow('Jetson Nano - Object Detection', frame)
        
        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("\nQuitting...")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed")
    print("Done!")


if __name__ == "__main__":
    main()
