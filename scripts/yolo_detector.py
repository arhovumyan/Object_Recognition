#!/home/aro/Documents/ObjectRec/.venv/bin/python3

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
        
        # Parameters (exposed to launch files)
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('confidence_threshold', 0.30)
        self.declare_parameter('phone_confidence_threshold', 0.20)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('imgsz', 960)  # inference size; larger helps small objects like phones
        
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.confidence_threshold = float(self.get_parameter('confidence_threshold').value)
        self.phone_confidence_threshold = float(self.get_parameter('phone_confidence_threshold').value)
        self.iou_threshold = float(self.get_parameter('iou_threshold').value)
        self.imgsz = int(self.get_parameter('imgsz').value)

        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Load YOLOv8s model with CPU-only configuration
        self.get_logger().info("Loading YOLOv8s model...")
        try:
            self.model = YOLO('yolov8s.pt')
            # Force CPU usage for YOLO
            self.model.to('cpu')
            self.get_logger().info("YOLOv8s model loaded successfully (CPU mode)")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLOv8s model: {str(e)}")
            return
        
        # Per-class thresholds (default higher, lower for phone)
        self.class_thresholds = {
            67: self.phone_confidence_threshold  # cell phone
        }
        
        # Get COCO class names
        self.class_names = self.model.names
        
        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
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
            
            # Run YOLO inference with CPU device (use configured thresholds and image size)
            results = self.model(cv_image, conf=self.confidence_threshold, iou=self.iou_threshold, imgsz=self.imgsz, device='cpu')
            
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
                        
                        # Threshold by class (fallback to general threshold)
                        is_phone = (class_id == 67)
                        detection_threshold = self.class_thresholds.get(class_id, self.confidence_threshold)
                        
                        if confidence > detection_threshold:
                            # Create detection message
                            detection = Detection2D()
                            detection.header = msg.header
                            
                            # Set bounding box (BoundingBox2D uses Pose2D center: x, y, theta)
                            detection.bbox.center.x = float((x1 + x2) / 2)
                            detection.bbox.center.y = float((y1 + y2) / 2)
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
                            elif not is_phone:
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
            
            # Display debug image in window
            cv2.imshow('YOLO Object Detection - Live Feed', debug_image)
            cv2.waitKey(1)
            
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
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
