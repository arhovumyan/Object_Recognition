#!/usr/bin/env python3

"""
Unified Live Object Detection with YOLOv8s
Works both as standalone script and ROS2 node.

Features:
- Real-time object detection with webcam
- GPU acceleration support (CUDA)
- ROS2 integration (optional)
- Optimized for performance
- Jetson Nano compatible

Usage:
    Standalone: python3 src/live_object_detection.py
    ROS2 Node:  ros2 run drone_object_recognition live_object_detection.py --ros-args
"""

import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
import argparse

# Import our MobileNetV3 classifier
try:
    from .mobilenet_classifier import MobileNetV3Classifier
    CLASSIFIER_AVAILABLE = True
except ImportError:
    try:
        from mobilenet_classifier import MobileNetV3Classifier
        CLASSIFIER_AVAILABLE = True
    except ImportError:
        print("Warning: MobileNetV3 classifier not available. Running YOLO-only mode.")
        CLASSIFIER_AVAILABLE = False

# Import detection logger
try:
    from .detection_logger import DetectionLogger, get_logger, cleanup_logger
    LOGGER_AVAILABLE = True
except ImportError:
    try:
        from detection_logger import DetectionLogger, get_logger, cleanup_logger
        LOGGER_AVAILABLE = True
    except ImportError:
        print("Warning: Detection logger not available. Running without logging.")
        LOGGER_AVAILABLE = False

# ROS2 imports (optional)
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
    from geometry_msgs.msg import Pose2D
    from std_msgs.msg import String, Bool
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


class ObjectDetectionSystem:
    """Unified object detection system that can work standalone or as ROS2 node."""
    
    def __init__(self, use_ros2=False, ros2_node=None, skip_camera=False, 
                 use_classifier=True, classifier_threshold=0.3, enable_logging=True):
        self.use_ros2 = use_ros2 and ROS2_AVAILABLE
        self.ros2_node = ros2_node
        self.use_classifier = use_classifier and CLASSIFIER_AVAILABLE
        self.classifier_threshold = classifier_threshold
        self.enable_logging = enable_logging and LOGGER_AVAILABLE
        
        # Initialize logger
        if self.enable_logging:
            self.logger = get_logger()
            print("Detection logging enabled")
        else:
            self.logger = None
        
        # Initialize YOLO model
        self.setup_model()
        
        # Initialize MobileNetV3 classifier
        if self.use_classifier:
            self.setup_classifier()
        
        # Target objects to highlight (for display purposes)
        self.target_objects = ['person', 'cell phone', 'laptop', 'mouse', 'tv', 
                              'bottle', 'cup', 'book', 'keyboard', 'chair']
        
        # YOLO classes to filter for MobileNetV3 classification
        self.yolo_filter_classes = ['person', 'backpack', 'umbrella', 'handbag', 'suitcase']
        
        # Camera setup (skip in ROS2 mode if we're subscribing to a topic)
        self.cap = None
        if not skip_camera:
            self.setup_camera()
        
        # ROS2 setup (if enabled)
        if self.use_ros2:
            self.setup_ros2()
        
        # Performance tracking
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.fps = 0
        
        # Frame skipping for performance
        self.frame_skip = 1 if self.use_gpu else 2
        self.frame_count = 0
        self.last_detections = []
        self.last_frame = None
    
    def setup_model(self):
        """Initialize YOLO model with GPU/CPU detection."""
        print("=" * 60)
        print("UNIFIED Live Object Detection with YOLOv8s")
        print("=" * 60)
        
        # Check if CUDA is available
        self.use_gpu = torch.cuda.is_available()
        
        if self.use_gpu:
            print(f"GPU Device detected: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
            # Check if GPU is compatible - test with a simple model inference
            try:
                # Create a small test tensor and try to use it on GPU
                test_tensor = torch.tensor([1.0]).cuda()
                
                # Try to load a small YOLO model on GPU as a compatibility test
                test_model = YOLO('yolov8n.pt')
                test_model.to('cuda')
                
                # Try a small inference test
                dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                test_result = test_model(dummy_frame, device='cuda', verbose=False, imgsz=320)
                
                # If we get here, GPU works
                del test_tensor
                del test_model
                del test_result
                torch.cuda.empty_cache()
                
                self.device = 'cuda'
                print("GPU compatibility check passed! Using GPU acceleration")
            except Exception as e:
                print(f"\nWarning: GPU detected but not compatible with PyTorch: {e}")
                print("This often happens when PyTorch is not compiled for your GPU architecture.")
                print("Falling back to CPU mode...")
                self.device = 'cpu'
                self.use_gpu = False
                torch.cuda.empty_cache()
        else:
            print("No GPU detected")
            self.device = 'cpu'
            self.use_gpu = False
        
        print(f"\nUsing device: {self.device.upper()}")
        
        # Load YOLO model - use YOLOv8s for GPU, YOLOv8n for CPU
        model_name = 'yolov8s.pt' if self.use_gpu else 'yolov8n.pt'
        
        # Look for model in models directory first
        candidate_paths = [
            os.path.join('..', 'models', model_name),  # ../models/yolov8s.pt
            os.path.join('models', model_name),        # models/yolov8s.pt
            os.path.join('..', model_name),            # ../yolov8s.pt
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
        self.model = YOLO(model_file)
        
        if self.use_gpu:
            try:
                self.model.to(self.device)
                print("Model moved to GPU")
            except Exception as e:
                print(f"Error moving model to GPU: {e}")
                print("Falling back to CPU...")
                self.device = 'cpu'
                self.use_gpu = False
                self.model = YOLO('yolov8n.pt')
        
        print("Model loaded successfully!")
        
        # Log system initialization
        if self.logger:
            self.logger._log_system_start()
    
    def setup_classifier(self):
        """Initialize MobileNetV3 classifier."""
        print("=" * 60)
        print("Initializing MobileNetV3 Classifier")
        print("=" * 60)
        
        try:
            # Look for fine-tuned model in models directory
            model_paths = [
                os.path.join('..', 'models', 'mobilenetv3_tent_mannequin.pth'),
                os.path.join('models', 'mobilenetv3_tent_mannequin.pth'),
                os.path.join('..', 'mobilenetv3_tent_mannequin.pth'),
                'mobilenetv3_tent_mannequin.pth'
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            # Initialize classifier
            self.classifier = MobileNetV3Classifier(model_path=model_path)
            
            if model_path:
                print(f"MobileNetV3: Using fine-tuned model from {model_path}")
            else:
                print("MobileNetV3: Using pre-trained ImageNet weights")
                print("MobileNetV3: Note - For best results, fine-tune on aerial tent/mannequin images")
            
            print("MobileNetV3: Classifier ready for inference")
            
            # Log classifier initialization
            if self.logger:
                self.logger._write_log({
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'event_type': 'classifier_init',
                    'classifier_available': True,
                    'model_path': model_path or 'ImageNet pre-trained',
                    'device': self.classifier.device if hasattr(self, 'classifier') else 'unknown'
                })
            
        except Exception as e:
            print(f"Error initializing MobileNetV3 classifier: {e}")
            print("Falling back to YOLO-only mode")
            self.use_classifier = False
            
            # Log classifier failure
            if self.logger:
                self.logger._write_log({
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'event_type': 'classifier_init_failed',
                    'error': str(e),
                    'fallback_mode': 'yolo_only'
                })
    
    def setup_camera(self):
        """Initialize camera."""
        print("Opening webcam...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            print("Trying alternative camera devices...")
            for i in range(1, 5):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    print(f"Successfully opened camera device {i}")
                    break
            else:
                print("Failed to open any camera device")
                return
        
        # Set camera properties for optimal performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Get actual camera resolution
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {width}x{height}")
        print()
        print("=" * 60)
        print("Detection running! Press 'q' to quit")
        print("=" * 60)
    
    def setup_ros2(self):
        """Setup ROS2 publishers and subscribers."""
        if not self.use_ros2 or not self.ros2_node:
            return
        
        # ROS2 publishers
        self.detection_pub = self.ros2_node.create_publisher(
            Detection2DArray, '/object_detections', 10)
        self.status_pub = self.ros2_node.create_publisher(
            String, '/recognition_status', 10)
        self.target_found_pub = self.ros2_node.create_publisher(
            Bool, '/target_object_found', 10)
        self.target_pos_pub = self.ros2_node.create_publisher(
            Pose2D, '/target_position', 10)
        
        # New publishers for two-stage pipeline
        self.target_type_pub = self.ros2_node.create_publisher(
            String, '/target_type', 10)
        self.payload_trigger_pub = self.ros2_node.create_publisher(
            Bool, '/payload_drop_trigger', 10)
        self.classification_status_pub = self.ros2_node.create_publisher(
            String, '/classification_status', 10)
        
        # CV bridge for image conversion
        self.bridge = CvBridge()
        
        self.ros2_node.get_logger().info('Object Detection System with ROS2 integration started')
    
    def filter_yolo_detections(self, results):
        """Filter YOLO detections to only include classes relevant for classification."""
        filtered_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Get class name
                    class_id = int(box.cls[0].cpu())
                    class_name = self.model.names[class_id]
                    
                    # Filter for relevant classes
                    if class_name in self.yolo_filter_classes:
                        # Get box coordinates and confidence
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                        confidence = float(box.conf[0].cpu())
                        
                        # Only include high-confidence detections
                        if confidence > 0.3:
                            filtered_detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_name': class_name,
                                'class_id': class_id
                            })
        
        return filtered_detections
    
    def process_detections(self, results, frame):
        """Process detection results with two-stage pipeline and handle ROS2 publishing."""
        detections_msg = None
        target_found = False
        target_position = None
        target_type = None
        
        if self.use_ros2:
            detections_msg = Detection2DArray()
            detections_msg.header.stamp = self.ros2_node.get_clock().now().to_msg()
            detections_msg.header.frame_id = "camera_frame"
        
        # Process with two-stage pipeline if classifier is available
        if self.use_classifier:
            # Filter YOLO detections for relevant classes
            filtered_detections = self.filter_yolo_detections(results)
            
            if filtered_detections:
                # Extract crops for classification
                crops = []
                for detection in filtered_detections:
                    crop = self.classifier.crop_from_bbox(frame, detection['bbox'])
                    crops.append(crop)
                
                # Classify crops with MobileNetV3
                classification_results = self.classifier.classify_crops(crops)
                
                # Process classification results
                for i, (detection, classification) in enumerate(zip(filtered_detections, classification_results)):
                    x1, y1, x2, y2 = detection['bbox']
                    yolo_conf = detection['confidence']
                    yolo_class = detection['class_name']
                    
                    # Get classification result
                    class_name = classification['class']
                    class_conf = classification['confidence']
                    
                    # Log classification result
                    if self.logger:
                        inference_time = 0.05  # Approximate inference time
                        self.logger.log_classification_result(
                            yolo_class, yolo_conf, class_name, class_conf,
                            [x1, y1, x2, y2], 
                            class_name in ['tent', 'mannequin'] and class_conf >= self.classifier_threshold,
                            class_name if class_name in ['tent', 'mannequin'] else None,
                            'water_bottle' if class_name == 'mannequin' else 'beacon' if class_name == 'tent' else None,
                            inference_time, self.frame_count, self.fps, self.device
                        )
                    
                    # Only process tent/mannequin detections above threshold
                    if class_name in ['tent', 'mannequin'] and class_conf >= self.classifier_threshold:
                        # This is a target!
                        target_found = True
                        
                        # Determine target type and payload
                        if class_name == 'mannequin':
                            target_type = 'mannequin'
                            payload_type = 'water_bottle'
                            color = (0, 255, 255)  # Yellow for mannequin
                        else:  # tent
                            target_type = 'tent'
                            payload_type = 'beacon'
                            color = (0, 255, 0)  # Green for tent
                        
                        thickness = 4
                        
                        # Calculate center position for payload drop
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        target_position = Pose2D() if self.use_ros2 else None
                        if target_position:
                            target_position.x = center_x
                            target_position.y = center_y
                            target_position.theta = class_conf  # Store confidence in theta
                        
                        # Draw thick bounding box for confirmed targets
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Draw classification label
                        label = f"{class_name.upper()} ({payload_type}) {class_conf:.2f}"
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2,
                            cv2.LINE_AA
                        )
                        
                        # Draw target center
                        cv2.circle(frame, (int(center_x), int(center_y)), 8, color, -1)
                        cv2.circle(frame, (int(center_x), int(center_y)), 12, (255, 255, 255), 2)
                        
                        # Log payload trigger
                        if self.logger:
                            self.logger.log_payload_trigger(
                                target_type, payload_type, class_conf,
                                center_x, center_y, self.frame_count, self.fps
                            )
                        
                        # ROS2 message creation for confirmed targets
                        if self.use_ros2 and detections_msg is not None:
                            detection_msg = Detection2D()
                            detection_msg.bbox.center.position.x = center_x
                            detection_msg.bbox.center.position.y = center_y
                            detection_msg.bbox.size_x = float(x2 - x1)
                            detection_msg.bbox.size_y = float(y2 - y1)
                            
                            hypothesis = ObjectHypothesisWithPose()
                            hypothesis.hypothesis.class_id = f"{class_name}_{payload_type}"
                            hypothesis.hypothesis.score = class_conf
                            detection_msg.results.append(hypothesis)
                            
                            detections_msg.detections.append(detection_msg)
                    else:
                        # Not a target, draw with normal styling
                        color = (128, 128, 128)  # Gray for filtered but not targets
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                        
                        label = f"{yolo_class}->{class_name} {class_conf:.2f}"
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1,
                            cv2.LINE_AA
                        )
            
            # Draw all other YOLO detections normally
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                        confidence = float(box.conf[0].cpu())
                        class_id = int(box.cls[0].cpu())
                        class_name = self.model.names[class_id]
                        
                        # Skip if already processed by classifier
                        if class_name in self.yolo_filter_classes:
                            continue
                        
                        # Draw other objects normally
                        color = (255, 0, 0)  # Blue for other objects
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                        
                        label = f"{class_name[:10]} {confidence:.1f}"
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1,
                            cv2.LINE_AA
                        )
        else:
            # Fallback to original YOLO-only processing
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                        confidence = float(box.conf[0].cpu())
                        class_id = int(box.cls[0].cpu())
                        class_name = self.model.names[class_id]
                        
                        color = (255, 0, 0)  # Blue for all objects in YOLO-only mode
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"{class_name[:10]} {confidence:.1f}"
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
        
        # Publish ROS2 messages
        if self.use_ros2:
            if detections_msg is not None:
                self.detection_pub.publish(detections_msg)
            
            self.target_found_pub.publish(Bool(data=target_found))
            
            if target_position:
                self.target_pos_pub.publish(target_position)
            
            # Publish target type for payload selection
            if target_type:
                target_type_msg = String()
                target_type_msg.data = target_type
                self.target_type_pub.publish(target_type_msg)
            
            # Publish payload drop trigger
            if target_found:
                trigger_msg = Bool()
                trigger_msg.data = True
                self.payload_trigger_pub.publish(trigger_msg)
            
            # Publish classification status
            if self.use_classifier:
                classification_msg = String()
                classification_msg.data = f"Pipeline: YOLO→MobileNetV3 | Threshold: {self.classifier_threshold}"
                self.classification_status_pub.publish(classification_msg)
            
            # Publish status
            status_msg = String()
            if self.use_classifier:
                status_msg.data = f"Two-stage: {len(detections_msg.detections) if detections_msg else 0} targets found"
            else:
                status_msg.data = f"YOLO-only: {len(detections_msg.detections) if detections_msg else 0} objects"
            self.status_pub.publish(status_msg)
        
        # Log performance update periodically
        if self.logger and self.frame_count % 100 == 0:
            self.logger.log_performance_update(self.fps, self.device, self.frame_count)
    
    def run_detection_cycle(self, frame):
        """Run one detection cycle."""
        self.frame_count += 1
        
        # Only run detection on every Nth frame
        if self.frame_count % self.frame_skip == 0:
            # Run YOLO detection with optimized settings
            inference_size = 640 if self.use_gpu else 320
            max_detections = 50 if self.use_gpu else 20
            
            try:
                results = self.model(
                    frame, 
                    conf=0.3,
                    iou=0.5,
                    max_det=max_detections,
                    verbose=False,
                    imgsz=inference_size,
                    half=False,
                    device=self.device
                )
            except (RuntimeError, Exception) as e:
                error_msg = str(e)
                if 'CUDA' in error_msg or 'no kernel image' in error_msg:
                    if self.use_gpu:
                        if self.use_ros2:
                            self.ros2_node.get_logger().error(f"GPU Error: {error_msg}")
                            self.ros2_node.get_logger().info("Switching to CPU mode...")
                        else:
                            print(f"\n\nGPU Error: {error_msg}")
                            print("Switching to CPU mode and reloading model...")
                        
                        self.device = 'cpu'
                        self.use_gpu = False
                        self.model = YOLO('yolov8n.pt')
                        self.frame_skip = 2
                        
                        if self.use_ros2:
                            self.ros2_node.get_logger().info("Model reloaded on CPU. Continuing detection...")
                        else:
                            print("Model reloaded on CPU. Continuing detection...\n")
                        
                        results = self.model(
                            frame, 
                            conf=0.3,
                            iou=0.5,
                            max_det=20,
                            verbose=False,
                            imgsz=320,
                            device='cpu'
                        )
                    else:
                        # Already on CPU, re-raise the error
                        raise
                else:
                    raise
            
            # Store results for next frames
            self.last_detections = results
            self.last_frame = frame.copy()
        else:
            # Reuse last detection results
            results = self.last_detections
            frame = self.last_frame if self.last_frame is not None else frame
        
        # Process results
        self.process_detections(results, frame)
        
        return frame
    
    def run_standalone(self):
        """Run standalone object detection (no ROS2)."""
        while True:
            # Read frame from webcam
            ret, frame = self.cap.read()
            
            if not ret:
                print("Error: Failed to grab frame")
                break
            
            # Run detection
            frame = self.run_detection_cycle(frame)
            
            # Calculate FPS
            self.fps_counter += 1
            if time.time() - self.fps_start_time >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            # Display FPS
            device_display = "GPU" if self.use_gpu else "CPU"
            fps_text = f"FPS: {self.fps} | Device: {device_display}"
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
            cv2.imshow('YOLOv8s Live Object Detection - UNIFIED', frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\nQuitting...")
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")
        print("Done!")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Cleanup logger
        if self.logger:
            self.logger.save_json_logs()
            self.logger.print_summary()


class ROS2ObjectDetectionNode(Node):
    """ROS2 Node wrapper for the object detection system."""
    
    def __init__(self):
        super().__init__('object_detection_node')
        
        # Don't open camera in ROS2 mode - subscribe to camera topic instead
        self.detection_system = ObjectDetectionSystem(use_ros2=True, ros2_node=self, skip_camera=True, enable_logging=True)
        
        # Subscribe to camera topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        
        # Performance tracking
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.fps = 0
        
        self.get_logger().info('ROS2 Object Detection Node started')
        self.get_logger().info('Subscribing to /camera/image_raw topic')
    
    def image_callback(self, msg):
        """Process incoming camera images from ROS2 topic."""
        try:
            # Convert ROS image to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Run detection cycle
            processed_frame = self.detection_system.run_detection_cycle(frame)
            
            # Calculate FPS
            self.fps_counter += 1
            if time.time() - self.fps_start_time >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            # Display FPS on frame
            device_display = "GPU" if self.detection_system.use_gpu else "CPU"
            fps_text = f"FPS: {self.fps} | Device: {device_display} | ROS2 Mode"
            cv2.putText(
                processed_frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            
            # Display the frame in a window
            cv2.imshow('ROS2 Object Detection - Live Feed', processed_frame)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')


def main():
    """Main function that handles both standalone and ROS2 modes."""
    parser = argparse.ArgumentParser(description='Unified Object Detection System')
    parser.add_argument('--ros-args', action='store_true', 
                       help='Run as ROS2 node')
    args, unknown = parser.parse_known_args()
    
    if args.ros_args and ROS2_AVAILABLE:
        # ROS2 mode
        print("Starting as ROS2 Node...")
        rclpy.init()
        node = ROS2ObjectDetectionNode()
        
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.detection_system.cleanup()
            node.destroy_node()
            # Protect against double shutdown
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception as e:
                pass  # Ignore shutdown errors
    else:
        # Standalone mode
        print("Starting in Standalone Mode...")
        detection_system = ObjectDetectionSystem(use_ros2=False)
        try:
            detection_system.run_standalone()
        except KeyboardInterrupt:
            pass
        finally:
            detection_system.cleanup()


if __name__ == "__main__":
    main()
