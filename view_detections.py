#!/usr/bin/env python3

"""
Simple image viewer for ROS2 image topics using OpenCV.
Displays the /yolo_debug_image topic with bounding boxes.

Usage:
    source /opt/ros/jazzy/setup.bash
    source install/setup.bash
    python3 view_detections.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class ImageViewer(Node):
    def __init__(self):
        super().__init__('image_viewer')
        
        self.bridge = CvBridge()
        
        # Subscribe to debug image topic
        self.subscription = self.create_subscription(
            Image,
            '/yolo_debug_image',
            self.image_callback,
            10
        )
        
        self.get_logger().info("Image Viewer started. Displaying /yolo_debug_image")
        self.get_logger().info("Press 'q' to quit")
        
    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Display image
            cv2.imshow('YOLO Detection - ROS2', cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Error displaying image: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    
    viewer = ImageViewer()
    
    try:
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
