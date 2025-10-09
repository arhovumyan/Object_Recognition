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
    
    def __init__(self, use_ros2=False, ros2_node=None, skip_camera=False):
        self.use_ros2 = use_ros2 and ROS2_AVAILABLE
        self.ros2_node = ros2_node
        
        # Initialize YOLO model
        self.setup_model()
        
        # Target objects to highlight
        self.target_objects = ['person', 'cell phone', 'laptop', 'mouse', 'tv', 
                              'bottle', 'cup', 'book', 'keyboard', 'chair']
        
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
        
        # CV bridge for image conversion
        self.bridge = CvBridge()
        
        self.ros2_node.get_logger().info('Object Detection System with ROS2 integration started')
    
    def process_detections(self, results, frame):
        """Process detection results and handle ROS2 publishing."""
        detections_msg = None
        target_found = False
        target_position = None
        
        if self.use_ros2:
            detections_msg = Detection2DArray()
            detections_msg.header.stamp = self.ros2_node.get_clock().now().to_msg()
            detections_msg.header.frame_id = "camera_frame"
        
        for result in results:
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                    
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu())
                    class_id = int(box.cls[0].cpu())
                    class_name = self.model.names[class_id]
                    
                    # Choose color based on whether it's a target object
                    if class_name in self.target_objects:
                        color = (0, 255, 0)  # Green for target objects
                        thickness = 3
                    else:
                        color = (255, 0, 0)  # Blue for other objects
                        thickness = 2
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Prepare label
                    label = f"{class_name[:10]} {confidence:.1f}"
                    
                    # Draw label
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
                    
                    # ROS2 message creation
                    if self.use_ros2 and detections_msg is not None:
                        detection = Detection2D()
                        detection.bbox.center.position.x = float((x1 + x2) / 2)
                        detection.bbox.center.position.y = float((y1 + y2) / 2)
                        detection.bbox.size_x = float(x2 - x1)
                        detection.bbox.size_y = float(y2 - y1)
                        
                        hypothesis = ObjectHypothesisWithPose()
                        hypothesis.hypothesis.class_id = class_name
                        hypothesis.hypothesis.score = confidence
                        detection.results.append(hypothesis)
                        
                        detections_msg.detections.append(detection)
                        
                        # Check if target object found
                        if class_name in self.target_objects and confidence > 0.5:
                            target_found = True
                            target_position = Pose2D()
                            target_position.x = detection.bbox.center.position.x
                            target_position.y = detection.bbox.center.position.y
                            target_position.theta = confidence
        
        # Publish ROS2 messages
        if self.use_ros2:
            if detections_msg is not None:
                self.detection_pub.publish(detections_msg)
            
            self.target_found_pub.publish(Bool(data=target_found))
            
            if target_position:
                self.target_pos_pub.publish(target_position)
            
            # Publish status
            status_msg = String()
            status_msg.data = f"Detected {len(detections_msg.detections) if detections_msg else 0} objects"
            self.status_pub.publish(status_msg)
    
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


class ROS2ObjectDetectionNode(Node):
    """ROS2 Node wrapper for the object detection system."""
    
    def __init__(self):
        super().__init__('object_detection_node')
        
        # Don't open camera in ROS2 mode - subscribe to camera topic instead
        self.detection_system = ObjectDetectionSystem(use_ros2=True, ros2_node=self, skip_camera=True)
        
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
