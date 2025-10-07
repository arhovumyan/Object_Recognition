#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import json
import csv
from datetime import datetime
from collections import defaultdict


class CameraRecorder(Node):
    """
    ROS2 Node that opens USB camera, publishes video stream, and records video with detection overlays.
    """

    def __init__(self):
        super().__init__('camera_recorder')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Camera parameters
        self.camera_device = self.declare_parameter('camera_device', '/dev/video0').value
        self.camera_width = self.declare_parameter('camera_width', 640).value
        self.camera_height = self.declare_parameter('camera_height', 480).value
        self.fps = self.declare_parameter('fps', 30).value
        self.record_video = self.declare_parameter('record_video', True).value
        
        # Detection logging
        self.detection_log = []
        self.detection_stats = defaultdict(int)
        self.session_start_time = datetime.now()
        
        # Initialize camera
        self.get_logger().info(f"Opening camera: {self.camera_device}")
        self.cap = cv2.VideoCapture(self.camera_device)
        
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera: {self.camera_device}")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Get actual camera properties
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Ensure we have a valid FPS (fallback to 30 if invalid)
        if actual_fps <= 0 or actual_fps > 120:
            actual_fps = 30.0
            self.get_logger().warn(f"Invalid FPS detected, using default: {actual_fps}")
        
        self.get_logger().info(f"Camera opened successfully:")
        self.get_logger().info(f"  Resolution: {actual_width}x{actual_height}")
        self.get_logger().info(f"  Actual FPS: {actual_fps}")
        
        # Store actual FPS for video recording
        self.actual_fps = actual_fps
        
        # Video recording setup
        self.video_writer = None
        self.recording_start_time = None
        
        if self.record_video:
            self.setup_video_recording(actual_width, actual_height, actual_fps)
        
        # Publishers
        self.image_pub = self.create_publisher(
            Image,
            '/camera/image_raw',
            10
        )
        
        # Subscribers for detection results
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/object_detections',
            self.detection_callback,
            10
        )
        
        # Current detection data
        self.current_detections = []
        self.current_frame = None
        
        # Timer for camera capture - use actual camera FPS for real-time speed
        self.timer_period = 1.0 / actual_fps  # Use actual camera FPS for real-time playback
        self.timer = self.create_timer(self.timer_period, self.capture_and_publish)
        
        # Statistics
        self.frame_count = 0
        self.start_time = self.get_clock().now()
        
        self.get_logger().info("Camera Recorder node initialized")

    def detection_callback(self, msg):
        """Callback for detection results."""
        self.current_detections = []
        
        for detection in msg.detections:
            if detection.results:
                result = detection.results[0]
                class_id = result.hypothesis.class_id
                confidence = result.hypothesis.score
                
                # Extract bounding box
                center_x = detection.bbox.center.position.x
                center_y = detection.bbox.center.position.y
                size_x = detection.bbox.size_x
                size_y = detection.bbox.size_y
                
                # Calculate bounding box coordinates
                x1 = int(center_x - size_x/2)
                y1 = int(center_y - size_y/2)
                x2 = int(center_x + size_x/2)
                y2 = int(center_y + size_y/2)
                
                detection_info = {
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'center': (int(center_x), int(center_y))
                }
                
                self.current_detections.append(detection_info)
                
                # Log detection
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.detection_log.append({
                    'timestamp': timestamp,
                    'frame_number': self.frame_count,
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': detection_info['bbox'],
                    'center': detection_info['center']
                })
                
                # Update statistics
                self.detection_stats[class_id] += 1

    def draw_detections(self, frame):
        """Draw detection overlays on the frame."""
        overlay_frame = frame.copy()
        
        # Colors for different object types
        colors = {
            'person': (0, 255, 0),      # Green
            'cell phone': (255, 0, 0),  # Blue
            'phone': (255, 0, 0),       # Blue
            'toothbrush': (0, 255, 255), # Yellow
            'mouse': (255, 0, 255),     # Magenta
            'hat': (0, 165, 255),       # Orange
        }
        
        for detection in self.current_detections:
            class_id = detection['class_id']
            confidence = detection['confidence']
            x1, y1, x2, y2 = detection['bbox']
            
            # Get color for this class
            color = colors.get(class_id, (255, 255, 255))  # Default white
            
            # Draw bounding box
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            label = f"{class_id}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(overlay_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(overlay_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return overlay_frame

    def setup_video_recording(self, width, height, fps):
        """Setup high-quality video recording to file."""
        try:
            # Create recordings directory
            recordings_dir = "recordings"
            if not os.path.exists(recordings_dir):
                os.makedirs(recordings_dir)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{recordings_dir}/detection_recording_{timestamp}.mp4"
            
            # Try high-quality codecs in order of preference
            codecs_to_try = [
                ('H264', cv2.VideoWriter_fourcc(*'H264'), {'quality': 100}),
                ('avc1', cv2.VideoWriter_fourcc(*'avc1'), {'quality': 100}),
                ('MJPG', cv2.VideoWriter_fourcc(*'MJPG'), {'quality': 95}),
                ('XVID', cv2.VideoWriter_fourcc(*'XVID'), {'quality': 90}),
                ('mp4v', cv2.VideoWriter_fourcc(*'mp4v'), {'quality': 85})
            ]
            
            self.video_writer = None
            for codec_name, fourcc, params in codecs_to_try:
                self.get_logger().info(f"Trying high-quality codec: {codec_name}")
                
                # Try with high quality settings
                self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                
                if self.video_writer.isOpened():
                    self.recording_start_time = self.get_clock().now()
                    self.get_logger().info(f"High-quality video recording started with {codec_name}: {filename}")
                    self.get_logger().info(f"  Resolution: {width}x{height}")
                    self.get_logger().info(f"  FPS: {fps}")
                    self.get_logger().info(f"  Quality: {params['quality']}%")
                    break
                else:
                    self.get_logger().warn(f"Failed to initialize video writer with {codec_name}")
                    self.video_writer = None
            
            if self.video_writer is None:
                self.get_logger().error("Failed to initialize video writer with any codec")
                
        except Exception as e:
            self.get_logger().error(f"Failed to setup video recording: {str(e)}")
            self.video_writer = None

    def capture_and_publish(self):
        """Capture frame from camera and publish it."""
        try:
            ret, frame = self.cap.read()
            
            if not ret:
                self.get_logger().warn("Failed to capture frame from camera")
                return
            
            # Increment frame count
            self.frame_count += 1
            
            # Store current frame for detection overlays
            self.current_frame = frame.copy()
            
            # Draw detection overlays on frame
            overlay_frame = self.draw_detections(frame)
            
            # Add frame info overlay
            self.add_frame_info_overlay(overlay_frame)
            
            # Convert OpenCV image to ROS message
            ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")  # Publish original frame
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = "camera_frame"
            
            # Publish image
            self.image_pub.publish(ros_image)
            
            # Record video with overlays if enabled
            if self.video_writer is not None and self.video_writer.isOpened():
                self.video_writer.write(overlay_frame)
            
            # Log statistics every 100 frames
            if self.frame_count % 100 == 0:
                elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
                actual_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Log detection statistics
                total_detections = sum(self.detection_stats.values())
                stats_text = ", ".join([f"{k}: {v}" for k, v in self.detection_stats.items()])
                
                self.get_logger().info(f"Frame {self.frame_count}, FPS: {actual_fps:.1f}, "
                                     f"Total detections: {total_detections}, Stats: {stats_text}")
                
        except Exception as e:
            self.get_logger().error(f"Error in capture_and_publish: {str(e)}")

    def add_frame_info_overlay(self, frame):
        """Add frame information overlay to the video."""
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add frame number
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(frame, frame_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add FPS
        elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        actual_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        fps_text = f"FPS: {actual_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add detection count
        detection_count = len(self.current_detections)
        detection_text = f"Detections: {detection_count}"
        cv2.putText(frame, detection_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def save_detection_logs(self):
        """Save comprehensive detection logs and statistics."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detection log as JSON
            if self.detection_log:
                json_file = f"recordings/detection_log_{timestamp}.json"
                with open(json_file, 'w') as f:
                    json.dump(self.detection_log, f, indent=2)
                self.get_logger().info(f"Detection log saved: {json_file}")
            
            # Save detection log as CSV
            if self.detection_log:
                csv_file = f"recordings/detection_log_{timestamp}.csv"
                with open(csv_file, 'w', newline='') as f:
                    if self.detection_log:
                        writer = csv.DictWriter(f, fieldnames=self.detection_log[0].keys())
                        writer.writeheader()
                        writer.writerows(self.detection_log)
                self.get_logger().info(f"Detection CSV saved: {csv_file}")
            
            # Save session summary
            summary_file = f"recordings/session_summary_{timestamp}.json"
            elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            session_summary = {
                'session_start': self.session_start_time.isoformat(),
                'session_end': datetime.now().isoformat(),
                'duration_seconds': elapsed_time,
                'total_frames': self.frame_count,
                'average_fps': avg_fps,
                'total_detections': len(self.detection_log),
                'detection_statistics': dict(self.detection_stats),
                'camera_settings': {
                    'device': self.camera_device,
                    'width': self.camera_width,
                    'height': self.camera_height,
                    'fps': self.fps
                }
            }
            
            with open(summary_file, 'w') as f:
                json.dump(session_summary, f, indent=2)
            self.get_logger().info(f"Session summary saved: {summary_file}")
            
            # Log final statistics
            self.get_logger().info(f"Session ended. Total frames: {self.frame_count}, Average FPS: {avg_fps:.1f}")
            self.get_logger().info(f"Total detections: {len(self.detection_log)}")
            for class_id, count in self.detection_stats.items():
                self.get_logger().info(f"  {class_id}: {count}")
                
        except Exception as e:
            self.get_logger().error(f"Error saving detection logs: {str(e)}")

    def destroy_node(self):
        """Cleanup when node is destroyed."""
        try:
            # Save detection logs and statistics
            self.save_detection_logs()
            
            # Release camera
            if self.cap is not None:
                self.cap.release()
            
            # Properly close video writer
            if self.video_writer is not None:
                self.video_writer.release()
                # Give a moment for the file to be properly written
                import time
                time.sleep(0.5)
                self.get_logger().info("Video recording saved and finalized")
            
        except Exception as e:
            self.get_logger().error(f"Error during cleanup: {str(e)}")
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    camera_recorder = CameraRecorder()
    
    try:
        rclpy.spin(camera_recorder)
    except KeyboardInterrupt:
        camera_recorder.get_logger().info("Shutting down camera recorder...")
    finally:
        camera_recorder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
