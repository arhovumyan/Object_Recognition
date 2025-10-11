#!/usr/bin/env python3

"""
Test YOLO Data Publisher
This script publishes mock YOLO detection results to test the YOLO detection system.
"""

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time


class TestYoloPublisher(Node):
    """Test publisher for YOLO detection data."""
    
    def __init__(self):
        super().__init__('test_yolo_publisher')
        
        # Get parameters
        self.declare_parameter('publish_rate', 1.0)
        self.declare_parameter('detect_person', True)
        self.declare_parameter('detect_car', False)
        self.declare_parameter('detect_bottle', True)
        
        self.publish_rate = self.get_parameter('publish_rate').value
        self.detect_person = self.get_parameter('detect_person').value
        self.detect_car = self.get_parameter('detect_car').value
        self.detect_bottle = self.get_parameter('detect_bottle').value
        
        # Publisher
        self.yolo_result_pub = self.create_publisher(
            Detection2DArray,
            'yolo_result',
            10
        )
        
        # CV Bridge for image processing
        self.bridge = CvBridge()
        
        # Timer for publishing test data
        self.timer = self.create_timer(self.publish_rate, self.publish_test_detections)
        
        # Counter for simulation
        self.detection_counter = 0
        
        self.get_logger().info('Test YOLO Publisher started')
        self.get_logger().info(f'Publish rate: {self.publish_rate} Hz')
        self.get_logger().info(f'Detecting: person={self.detect_person}, car={self.detect_car}, bottle={self.detect_bottle}')
    
    def publish_test_detections(self):
        """Publish mock YOLO detection results."""
        self.detection_counter += 1
        
        # Create detection message
        detection_array = Detection2DArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = "camera_frame"
        
        # Simulate different detection scenarios
        if self.detection_counter % 10 == 0:  # Every 10th iteration, no detections
            self.get_logger().info('No objects detected (simulated)')
        elif self.detection_counter % 5 == 0 and self.detect_person:  # Every 5th iteration, detect person
            detection = self.create_person_detection()
            detection_array.detections.append(detection)
            self.get_logger().info('Detected: PERSON (simulated)')
        elif self.detection_counter % 7 == 0 and self.detect_car:  # Every 7th iteration, detect car
            detection = self.create_car_detection()
            detection_array.detections.append(detection)
            self.get_logger().info('Detected: CAR (simulated)')
        elif self.detection_counter % 3 == 0 and self.detect_bottle:  # Every 3rd iteration, detect bottle
            detection = self.create_bottle_detection()
            detection_array.detections.append(detection)
            self.get_logger().info('Detected: BOTTLE (simulated)')
        
        # Publish detection results
        self.yolo_result_pub.publish(detection_array)
    
    def create_person_detection(self):
        """Create a mock person detection."""
        detection = Detection2D()
        
        # Bounding box (center of image with some offset)
        detection.bbox.center.position.x = 320.0 + np.random.uniform(-50, 50)
        detection.bbox.center.position.y = 240.0 + np.random.uniform(-30, 30)
        detection.bbox.size_x = 100.0 + np.random.uniform(-20, 20)
        detection.bbox.size_y = 200.0 + np.random.uniform(-40, 40)
        
        # Object hypothesis (person = class 0 in COCO dataset)
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = "0"  # person
        hypothesis.hypothesis.score = 0.8 + np.random.uniform(-0.1, 0.1)
        detection.results.append(hypothesis)
        
        return detection
    
    def create_car_detection(self):
        """Create a mock car detection."""
        detection = Detection2D()
        
        # Bounding box
        detection.bbox.center.position.x = 320.0 + np.random.uniform(-100, 100)
        detection.bbox.center.position.y = 240.0 + np.random.uniform(-50, 50)
        detection.bbox.size_x = 150.0 + np.random.uniform(-30, 30)
        detection.bbox.size_y = 80.0 + np.random.uniform(-20, 20)
        
        # Object hypothesis (car = class 2 in COCO dataset)
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = "2"  # car
        hypothesis.hypothesis.score = 0.7 + np.random.uniform(-0.1, 0.1)
        detection.results.append(hypothesis)
        
        return detection
    
    def create_bottle_detection(self):
        """Create a mock bottle detection."""
        detection = Detection2D()
        
        # Bounding box
        detection.bbox.center.position.x = 320.0 + np.random.uniform(-80, 80)
        detection.bbox.center.position.y = 240.0 + np.random.uniform(-40, 40)
        detection.bbox.size_x = 40.0 + np.random.uniform(-10, 10)
        detection.bbox.size_y = 80.0 + np.random.uniform(-20, 20)
        
        # Object hypothesis (bottle = class 39 in COCO dataset)
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = "39"  # bottle
        hypothesis.hypothesis.score = 0.6 + np.random.uniform(-0.1, 0.1)
        detection.results.append(hypothesis)
        
        return detection


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    
    try:
        node = TestYoloPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
