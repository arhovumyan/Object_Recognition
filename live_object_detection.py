#!/usr/bin/env python3

"""
Live Object Detection with YOLOv8s (OPTIMIZED)
Uses webcam for real-time object detection and displays results.

Optimizations:
- Lower resolution for faster processing
- Half precision (FP16) for GPU
- Lower confidence threshold
- Optimized inference settings
- Reduced frame buffer

Usage:
    python3 live_object_detection.py
    
Press 'q' to quit
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch


def main():
    """Run live object detection on webcam feed."""
    print("=" * 60)
    print("OPTIMIZED Live Object Detection with YOLOv8s")
    print("=" * 60)
    
    # Check if CUDA is available
    use_gpu = torch.cuda.is_available()
    
    if use_gpu:
        print(f"GPU Device detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Check if GPU is compatible
        try:
            # Test GPU with a simple operation
            test_tensor = torch.tensor([1.0]).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            device = 'cuda'
            print("GPU compatibility check passed! Using GPU acceleration")
        except Exception as e:
            print(f"\nWarning: GPU detected but not compatible with PyTorch: {e}")
            print("Falling back to CPU mode...")
            print("(To use GPU, update PyTorch: pip install --upgrade torch torchvision)")
            device = 'cpu'
            use_gpu = False
    else:
        print("No GPU detected")
        device = 'cpu'
        use_gpu = False
    
    print(f"\nUsing device: {device.upper()}")
    
    # Load YOLO model - use YOLOv8n for CPU, YOLOv8s for GPU
    model_file = 'yolov8s.pt' if use_gpu else 'yolov8n.pt'
    print(f"Loading {model_file} model...")
    model = YOLO(model_file)
    
    if use_gpu:
        try:
            model.to(device)
            print("Model moved to GPU")
        except Exception as e:
            print(f"Error moving model to GPU: {e}")
            print("Falling back to CPU...")
            device = 'cpu'
            use_gpu = False
            # Reload with nano model for CPU
            model = YOLO('yolov8n.pt')
    
    print("Model loaded successfully!")
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        print("Trying alternative camera devices...")
        for i in range(1, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Successfully opened camera device {i}")
                break
        else:
            print("Failed to open any camera device")
            return
    
    # Set camera properties for optimal performance
    # Lower resolution = faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced from 1280
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduced from 720
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPEG for faster capture
    
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
    
    # Skip frame counter for performance
    frame_skip = 1 if use_gpu else 2  # Process every frame with GPU, every 2nd with CPU
    frame_count = 0
    
    # Store last detection results for skipped frames
    last_detections = []
    last_frame = None
    
    # Track if we've successfully run GPU inference
    gpu_working = use_gpu
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab frame")
            break
        
        frame_count += 1
        
        # Only run detection on every Nth frame
        if frame_count % frame_skip == 0:
            # Run YOLO detection with optimized settings
            inference_size = 640 if gpu_working else 320
            max_detections = 50 if gpu_working else 20
            
            try:
                results = model(
                    frame, 
                    conf=0.3,           # Confidence threshold
                    iou=0.5,            # IOU threshold for NMS
                    max_det=max_detections,
                    verbose=False,
                    imgsz=inference_size,
                    half=False,         # Disable half precision (not compatible)
                    device=device
                )
            except RuntimeError as e:
                if 'CUDA' in str(e) and gpu_working:
                    # GPU error occurred, fall back to CPU
                    print(f"\n\nGPU Error: {e}")
                    print("Switching to CPU mode and reloading model...")
                    device = 'cpu'
                    gpu_working = False
                    use_gpu = False
                    
                    # Reload model on CPU with nano version
                    model = YOLO('yolov8n.pt')
                    frame_skip = 2  # Process every 2nd frame on CPU
                    print("Model reloaded on CPU. Continuing detection...\n")
                    
                    # Try again with CPU
                    results = model(
                        frame, 
                        conf=0.3,
                        iou=0.5,
                        max_det=20,
                        verbose=False,
                        imgsz=320,
                        device='cpu'
                    )
                else:
                    raise
            
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
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                    
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu())
                    class_id = int(box.cls[0].cpu())
                    class_name = model.names[class_id]
                    
                    # Choose color based on whether it's a target object
                    if class_name in target_objects:
                        color = (0, 255, 0)  # Green for target objects
                        thickness = 3
                    else:
                        color = (255, 0, 0)  # Blue for other objects
                        thickness = 2
                    
                    # Draw bounding box (faster with thinner lines)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Prepare label (shorter for speed)
                    label = f"{class_name[:10]} {confidence:.1f}"
                    
                    # Simple text without background for speed
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
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
        
        # Display FPS with performance info
        device_display = "GPU" if gpu_working else "CPU"
        fps_text = f"FPS: {fps} | Device: {device_display}"
        cv2.putText(
            frame,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        
        # Display frame
        cv2.imshow('YOLOv8s Live Object Detection - OPTIMIZED', frame)
        
        # Check for quit key (non-blocking with waitKey(1))
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
