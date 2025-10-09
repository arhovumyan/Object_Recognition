#!/usr/bin/env python3

"""
ROS2 Camera Publisher Node
Publishes camera feed to ROS2 topics for other nodes to consume.

Usage:
    python3 src/camera_publisher_node.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os


class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')
        
        # Create publisher for camera feed
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        
        # CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Initialize camera
        self.setup_camera()
        
        # Create timer for publishing frames
        self.timer = self.create_timer(0.033, self.publish_frame)  # ~30 FPS
        
        self.get_logger().info('Camera Publisher Node started')
    
    def setup_camera(self):
        """Initialize camera"""
        # Try different camera indices
        camera_indices = [0, 1, 2]
        self.cap = None
        
        for i in camera_indices:
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                self.get_logger().info(f'Successfully opened camera device {i}')
                break
        
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().error('Could not open any camera device')
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Get actual camera resolution
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f'Camera resolution: {width}x{height}')
    
    def publish_frame(self):
        """Publish camera frame to ROS2 topic"""
        if self.cap is None or not self.cap.isOpened():
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to grab frame')
            return
        
        try:
            # Convert OpenCV image to ROS2 message
            image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.header.frame_id = "camera_frame"
            
            # Publish the image
            self.image_pub.publish(image_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing frame: {e}')
    
    def destroy_node(self):
        """Cleanup when node is destroyed"""
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # Protect against double shutdown
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            pass  # Ignore shutdown errors


if __name__ == '__main__':
    main()
