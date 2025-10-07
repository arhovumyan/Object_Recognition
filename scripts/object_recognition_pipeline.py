#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge
import cv2
import numpy as np


class ObjectRecognitionPipeline(Node):
    """
    Main ROS2 Node that coordinates YOLO detection and MobileNet classification.
    Provides high-level control and status reporting for the drone object recognition system.
    """

    def __init__(self):
        super().__init__('object_recognition_pipeline')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # State variables
        self.target_objects = ['toothbrush', 'mouse', 'phone', 'hat']
        self.current_target = 'phone'  # Default target
        self.target_found = False
        self.last_detection_time = None
        
        # Subscribers
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/object_detections',
            self.detection_callback,
            10
        )
        
        self.classification_sub = self.create_subscription(
            String,
            '/object_classification',
            self.classification_callback,
            10
        )
        
        self.target_found_sub = self.create_subscription(
            Bool,
            '/target_object_found',
            self.target_found_callback,
            10
        )
        
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Publishers
        self.status_pub = self.create_publisher(
            String,
            '/recognition_status',
            10
        )
        
        self.target_position_pub = self.create_publisher(
            Pose2D,
            '/target_position',
            10
        )
        
        self.command_pub = self.create_subscription(
            String,
            '/recognition_command',
            self.command_callback,
            10
        )
        
        # Timer for status updates
        self.status_timer = self.create_timer(1.0, self.publish_status)
        
        # Latest data
        self.latest_image = None
        self.latest_detections = None
        self.latest_classification = None
        
        self.get_logger().info("Object Recognition Pipeline initialized")
        self.get_logger().info(f"Current target: {self.current_target}")
        self.get_logger().info(f"Available targets: {', '.join(self.target_objects)}")

    def command_callback(self, msg):
        """
        Handle commands to change target or control the system.
        """
        command = msg.data.strip().lower()
        
        if command in self.target_objects:
            self.current_target = command
            self.get_logger().info(f"Target changed to: {self.current_target}")
        elif command == "start":
            self.get_logger().info("Recognition system started")
        elif command == "stop":
            self.get_logger().info("Recognition system stopped")
        elif command == "status":
            self.publish_status()

    def image_callback(self, msg):
        """
        Store latest image for processing.
        """
        self.latest_image = msg

    def detection_callback(self, msg):
        """
        Process YOLO detection results.
        """
        self.latest_detections = msg
        self.last_detection_time = self.get_clock().now()

    def classification_callback(self, msg):
        """
        Process MobileNet classification results.
        """
        self.latest_classification = msg.data

    def target_found_callback(self, msg):
        """
        Handle target found notifications.
        """
        self.target_found = msg.data
        
        if self.target_found and self.latest_detections:
            # Publish target position for drone control
            self.publish_target_position()

    def publish_target_position(self):
        """
        Publish the position of the found target object.
        """
        if not self.latest_detections or not self.latest_detections.detections:
            return
        
        try:
            # Find the most confident detection that matches our target
            best_detection = None
            best_confidence = 0.0
            
            for detection in self.latest_detections.detections:
                if detection.results:
                    confidence = detection.results[0].hypothesis.score
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_detection = detection
            
            if best_detection:
                position = Pose2D()
                position.x = best_detection.bbox.center.x
                position.y = best_detection.bbox.center.y
                position.theta = best_confidence  # Use theta to store confidence
                
                self.target_position_pub.publish(position)
                self.get_logger().info(f"Target position published: ({position.x:.1f}, {position.y:.1f}) confidence: {best_confidence:.3f}")
        
        except Exception as e:
            self.get_logger().error(f"Failed to publish target position: {str(e)}")

    def publish_status(self):
        """
        Publish system status information.
        """
        try:
            status_msg = String()
            
            # Build status string
            status_parts = [
                f"Target: {self.current_target}",
                f"Found: {'Yes' if self.target_found else 'No'}",
                f"Detections: {len(self.latest_detections.detections) if self.latest_detections else 0}",
                f"Classification: {self.latest_classification if self.latest_classification else 'None'}"
            ]
            
            status_msg.data = " | ".join(status_parts)
            self.status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish status: {str(e)}")

    def get_system_info(self):
        """
        Get comprehensive system information.
        """
        info = {
            'current_target': self.current_target,
            'available_targets': self.target_objects,
            'target_found': self.target_found,
            'detection_count': len(self.latest_detections.detections) if self.latest_detections else 0,
            'latest_classification': self.latest_classification,
            'has_image': self.latest_image is not None,
            'last_detection_time': self.last_detection_time
        }
        return info


def main(args=None):
    rclpy.init(args=args)
    
    pipeline = ObjectRecognitionPipeline()
    
    try:
        rclpy.spin(pipeline)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
