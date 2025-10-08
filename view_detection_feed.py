#!/usr/bin/env python3

"""
Simple viewer script to display the camera feed and detection results.
This script subscribes to the ROS2 topics and displays them using OpenCV.
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class DetectionViewer(Node):
    def __init__(self):
        super().__init__('detection_viewer')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Subscribe to camera feed
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10
        )
        
        # Subscribe to YOLO debug image
        self.yolo_sub = self.create_subscription(
            Image,
            '/yolo_debug_image',
            self.yolo_callback,
            10
        )
        
        # Subscribe to classification debug image
        self.classification_sub = self.create_subscription(
            Image,
            '/classification_debug_image',
            self.classification_callback,
            10
        )
        
        # Store latest images
        self.latest_camera_image = None
        self.latest_yolo_image = None
        self.latest_classification_image = None
        
        self.get_logger().info("Detection Viewer initialized")
        self.get_logger().info("Press 'q' to quit, 'c' for camera, 'y' for YOLO, 'm' for MobileNet")

    def camera_callback(self, msg):
        try:
            self.latest_camera_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting camera image: {e}")

    def yolo_callback(self, msg):
        try:
            self.latest_yolo_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting YOLO image: {e}")

    def classification_callback(self, msg):
        try:
            self.latest_classification_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting classification image: {e}")

    def display_images(self):
        """Display available images in separate windows."""
        while rclpy.ok():
            # Display camera feed
            if self.latest_camera_image is not None:
                cv2.imshow('Camera Feed', self.latest_camera_image)
            
            # Display YOLO detections
            if self.latest_yolo_image is not None:
                cv2.imshow('YOLO Object Detection', self.latest_yolo_image)
            
            # Display MobileNet classifications
            if self.latest_classification_image is not None:
                cv2.imshow('MobileNet Classification', self.latest_classification_image)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and self.latest_camera_image is not None:
                cv2.imshow('Camera Feed (Full)', self.latest_camera_image)
            elif key == ord('y') and self.latest_yolo_image is not None:
                cv2.imshow('YOLO Detections (Full)', self.latest_yolo_image)
            elif key == ord('m') and self.latest_classification_image is not None:
                cv2.imshow('MobileNet Classification (Full)', self.latest_classification_image)
            
            # Process ROS2 callbacks
            rclpy.spin_once(self, timeout_sec=0.01)


def main(args=None):
    rclpy.init(args=args)
    
    viewer = DetectionViewer()
    
    try:
        viewer.display_images()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        viewer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
