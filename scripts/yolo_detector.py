#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import os


class YOLODetector(Node):
    """
    ROS2 Node for YOLOv8s object detection.
    Detects objects in camera feed and publishes detection results.
    """

    def __init__(self):
        super().__init__('yolo_detector')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Load YOLOv8s model
        self.get_logger().info("Loading YOLOv8s model...")
        try:
            self.model = YOLO('yolov8s.pt')
            self.get_logger().info("YOLOv8s model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLOv8s model: {str(e)}")
            return
        
        # Define target classes - focus on phone detection
        # YOLO COCO classes that might match our targets
        self.target_classes = {
            'phone': 67,       # cell phone - our main target
            'person': 0,       # person (for hat detection)
            'laptop': 63,      # laptop (for mouse detection)
            'tv': 62,          # tv (additional objects)
            'remote': 74       # remote (additional objects)
        }
        
        # Phone-specific detection settings
        self.phone_confidence_threshold = 0.3  # Lower threshold for phone detection
        
        # Get COCO class names
        self.class_names = self.model.names
        
        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/object_detections',
            10
        )
        
        self.debug_image_pub = self.create_publisher(
            Image,
            '/yolo_debug_image',
            10
        )
        
        # Detection parameters - optimized for real-time phone detection
        self.confidence_threshold = 0.3  # Lower threshold for more detections
        self.iou_threshold = 0.45
        self.phone_detection_count = 0
        self.total_detection_count = 0
        
        self.get_logger().info("YOLO Detector node initialized")

    def image_callback(self, msg):
        """
        Process incoming camera image and detect objects.
        """
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Run YOLO inference
            results = self.model(cv_image, conf=self.confidence_threshold, iou=self.iou_threshold)
            
            # Process detections
            detections = Detection2DArray()
            detections.header = msg.header
            
            # Create debug image with bounding boxes
            debug_image = cv_image.copy()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box information
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.class_names[class_id]
                        
                        # Check if this is a target class or potentially relevant
                        is_target = self.is_target_object(class_name, class_id)
                        is_phone = (class_id == 67)  # cell phone class
                        
                        # Lower threshold for phone detection, higher for others
                        detection_threshold = self.phone_confidence_threshold if is_phone else 0.5
                        
                        if is_target and confidence > detection_threshold:
                            # Create detection message
                            detection = Detection2D()
                            detection.header = msg.header
                            
                            # Set bounding box
                            detection.bbox.center.position.x = float((x1 + x2) / 2)
                            detection.bbox.center.position.y = float((y1 + y2) / 2)
                            detection.bbox.size_x = float(x2 - x1)
                            detection.bbox.size_y = float(y2 - y1)
                            
                            # Set hypothesis
                            hypothesis = ObjectHypothesisWithPose()
                            hypothesis.hypothesis.class_id = str(class_id)
                            hypothesis.hypothesis.score = float(confidence)
                            detection.results.append(hypothesis)
                            
                            detections.detections.append(detection)
                            
                            # Update detection counters
                            self.total_detection_count += 1
                            if is_phone:
                                self.phone_detection_count += 1
                            
                            # Draw on debug image with special highlighting for phones
                            if is_phone:
                                color = (0, 255, 255)  # Yellow for phones
                                thickness = 3
                            elif is_target:
                                color = (0, 255, 0)    # Green for other targets
                                thickness = 2
                            else:
                                color = (255, 0, 0)    # Blue for others
                                thickness = 1
                                
                            cv2.rectangle(debug_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                            
                            # Add text with detection info
                            text = f"{class_name}: {confidence:.2f}"
                            if is_phone:
                                text = f"PHONE: {confidence:.2f}"
                            
                            cv2.putText(debug_image, text, 
                                      (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Publish detections
            self.detection_pub.publish(detections)
            
            # Log detection statistics every 50 frames
            if self.total_detection_count > 0 and self.total_detection_count % 50 == 0:
                phone_percentage = (self.phone_detection_count / self.total_detection_count) * 100
                self.get_logger().info(f"Detection stats - Total: {self.total_detection_count}, Phones: {self.phone_detection_count} ({phone_percentage:.1f}%)")
            
            # Publish debug image
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
                debug_msg.header = msg.header
                self.debug_image_pub.publish(debug_msg)
            except Exception as e:
                self.get_logger().warn(f"Failed to publish debug image: {str(e)}")
                
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

    def is_target_object(self, class_name, class_id):
        """
        Check if detected object is one of our target objects.
        Focus on phone detection but include other relevant objects.
        """
        # Primary target: cell phone (class_id 67)
        if class_id == 67:  # cell phone
            return True
            
        # Secondary targets for comprehensive detection
        target_names = ['person', 'laptop', 'tv', 'remote']
        if class_name in target_names:
            return True
            
        return False


def main(args=None):
    rclpy.init(args=args)
    
    detector = YOLODetector()
    
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
