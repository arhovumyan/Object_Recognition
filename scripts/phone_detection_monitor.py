#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np


class PhoneDetectionMonitor(Node):
    """
    ROS2 Node to monitor and display real-time phone detection results.
    Shows detection statistics and live video feed with phone highlighting.
    """

    def __init__(self):
        super().__init__('phone_detection_monitor')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Statistics
        self.phone_detections = 0
        self.total_detections = 0
        self.total_frames = 0
        self.start_time = self.get_clock().now()
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
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
        
        # Store latest data
        self.latest_image = None
        self.latest_detections = None
        self.latest_classification = ""
        self.phone_found = False
        
        # Display settings
        self.show_display = True
        self.display_width = 800
        self.display_height = 600
        
        self.get_logger().info("Phone Detection Monitor initialized")
        self.get_logger().info("Press 'q' to quit, 's' to toggle statistics")

    def image_callback(self, msg):
        """Store latest camera image."""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.total_frames += 1
            self.update_display()
        except Exception as e:
            self.get_logger().warn(f"Failed to process image: {str(e)}")

    def detection_callback(self, msg):
        """Process detection results."""
        self.latest_detections = msg
        self.total_detections += len(msg.detections)

    def classification_callback(self, msg):
        """Process classification results."""
        self.latest_classification = msg.data

    def target_found_callback(self, msg):
        """Process target found notifications."""
        if msg.data:
            self.phone_detections += 1

    def update_display(self):
        """Update the display with current detection results."""
        if self.latest_image is None:
            return
        
        try:
            # Create display image
            display_image = self.latest_image.copy()
            
            # Resize for display
            display_image = cv2.resize(display_image, (self.display_width, self.display_height))
            
            # Add detection overlays
            if self.latest_detections:
                for detection in self.latest_detections.detections:
                    # Convert detection coordinates to display coordinates
                    center_x = int(detection.bbox.center.x * self.display_width / self.latest_image.shape[1])
                    center_y = int(detection.bbox.center.y * self.display_height / self.latest_image.shape[0])
                    size_x = int(detection.bbox.size_x * self.display_width / self.latest_image.shape[1])
                    size_y = int(detection.bbox.size_y * self.display_height / self.latest_image.shape[0])
                    
                    x1 = int(center_x - size_x/2)
                    y1 = int(center_y - size_y/2)
                    x2 = int(center_x + size_x/2)
                    y2 = int(center_y + size_y/2)
                    
                    # Draw bounding box
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Add statistics overlay
            self.add_statistics_overlay(display_image)
            
            # Add classification text
            if self.latest_classification:
                cv2.putText(display_image, f"Classification: {self.latest_classification}", 
                          (10, self.display_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add phone detection indicator
            if self.phone_found:
                cv2.putText(display_image, "📱 PHONE DETECTED!", 
                          (self.display_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 3)
            
            # Display image
            cv2.imshow('Phone Detection Monitor', display_image)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("Quitting monitor...")
                cv2.destroyAllWindows()
                rclpy.shutdown()
            elif key == ord('s'):
                self.show_statistics()
                
        except Exception as e:
            self.get_logger().warn(f"Display update failed: {str(e)}")

    def add_statistics_overlay(self, image):
        """Add statistics overlay to the image."""
        try:
            # Calculate FPS
            elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            fps = self.total_frames / elapsed_time if elapsed_time > 0 else 0
            
            # Calculate detection rate
            detection_rate = (self.phone_detections / self.total_frames * 100) if self.total_frames > 0 else 0
            
            # Add statistics text
            stats_text = [
                f"FPS: {fps:.1f}",
                f"Frames: {self.total_frames}",
                f"Detections: {self.total_detections}",
                f"Phones: {self.phone_detections}",
                f"Phone Rate: {detection_rate:.1f}%"
            ]
            
            # Draw background rectangle
            cv2.rectangle(image, (10, 10), (300, 120), (0, 0, 0), -1)
            cv2.rectangle(image, (10, 10), (300, 120), (255, 255, 255), 2)
            
            # Draw statistics text
            for i, text in enumerate(stats_text):
                cv2.putText(image, text, (20, 35 + i * 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        except Exception as e:
            self.get_logger().warn(f"Statistics overlay failed: {str(e)}")

    def show_statistics(self):
        """Print detailed statistics."""
        elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        fps = self.total_frames / elapsed_time if elapsed_time > 0 else 0
        detection_rate = (self.phone_detections / self.total_frames * 100) if self.total_frames > 0 else 0
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("PHONE DETECTION STATISTICS")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Runtime: {elapsed_time:.1f} seconds")
        self.get_logger().info(f"FPS: {fps:.1f}")
        self.get_logger().info(f"Total Frames: {self.total_frames}")
        self.get_logger().info(f"Total Detections: {self.total_detections}")
        self.get_logger().info(f"Phone Detections: {self.phone_detections}")
        self.get_logger().info(f"Phone Detection Rate: {detection_rate:.1f}%")
        self.get_logger().info("=" * 50)


def main(args=None):
    rclpy.init(args=args)
    
    monitor = PhoneDetectionMonitor()
    
    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
