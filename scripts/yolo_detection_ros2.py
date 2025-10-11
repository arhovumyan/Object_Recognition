#!/usr/bin/env python3

"""
ROS2 Version of YOLO Detection System for Drone Flight Control
Converted from ROS1 to ROS2 with async service calls and modern ROS2 patterns.
"""

import rclpy
from rclpy.node import Node
from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import Detection2D
from mavros_msgs.srv import (
    CommandLong,
    SetMode,
    WaypointSetCurrent,
    WaypointPull,
)
from mavros_msgs.msg import WaypointReached, VFR_HUD, StatusText
from geometry_msgs.msg import Pose2D
import time
import cv2
import math
import sys
import os
import subprocess
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, NavSatFix
from cv_bridge import CvBridge
from collections import deque
import ament_index_python

# Color codes for terminal output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# Flight parameters
ALT = 20  # in meters (this is ~55 ft)

# Camera field of view (degrees)
HFOV = 68.75
VFOV = 53.13


class ServoController:
    """Simple servo controller for payload dropping."""
    
    def __init__(self):
        self.node = None
    
    def set_node(self, node):
        self.node = node
    
    def run_sequence(self):
        """Run the servo sequence for payload dropping."""
        if self.node:
            self.node.get_logger().info("Running servo sequence for payload drop")
            # Add your servo control logic here
            time.sleep(2)  # Simulate servo movement
            self.node.get_logger().info("Payload dropped successfully")


class Detected_Object_Waypoints:
    def __init__(self):
        self.detected_objects = deque()  # Queue to store detected objects and their GPS waypoints

    def add_object(self, object_name, lat, long, alt, index):
        """
        Adds a detected object with its name and calculated GPS coordinates to the queue.
        """
        if len(self.detected_objects) >= 4:
            return

        detected_object = {
            "name": object_name,
            "latitude": lat,
            "longitude": long,
            "alt": alt,
            "index": index,
        }
        self.detected_objects.append(detected_object)
        print(f"Object '{object_name}' added at LAT: {lat}, LONG: {long}, ALT: {alt}")

    def get_detected_objects(self):
        """
        Returns the queue of detected objects with their GPS waypoints.
        """
        return self.detected_objects

    def rotate_waypoints(self, rotate=-1):
        if self.detected_objects:
            self.detected_objects.rotate(rotate)

    def clear_queue(self):
        return self.detected_objects.clear()


class YoloResultSubscriber(Node):

    def __init__(self):
        super().__init__('yolo_result_subscriber')
        
        # Initialize components
        self.detected_object_waypoints = Detected_Object_Waypoints()
        self.last_before_rtl = 0
        self.next_after_takeoff = 0
        self.takeoff_index = 0
        self.rtl_index = 0
        self.lap = 0

        self.servo_controller = ServoController()
        self.servo_controller.set_node(self)

        # Subscribers
        self.subscription = self.create_subscription(
            YoloResult,
            'yolo_result',
            self.callback,
            1
        )
        
        self.waypoint_reached_sub = self.create_subscription(
            WaypointReached,
            '/mavros/mission/reached',
            self.update_waypoint_reached,
            10
        )
        
        self.vfr_hud_sub = self.create_subscription(
            VFR_HUD,
            '/mavros/vfr_hud',
            self.speed_cb,
            10
        )
        
        self.status_text_sub = self.create_subscription(
            StatusText,
            '/mavros/statustext/recv',
            self.restart_callback,
            10
        )
        
        self.yolo_image_sub = self.create_subscription(
            Image,
            '/yolo_image',
            self.yolo_image_callback,
            1
        )
        
        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/mavros/global_position/global',
            self.geofence_check,
            10
        )

        # Publishers
        self.status_pub = self.create_publisher(
            StatusText,
            '/mavros/statustext/send',
            10
        )
        self.detected_photo_pub = self.create_publisher(
            Bool,
            '/camera/object_detected',
            10
        )

        # Service clients
        self.set_wp_client = self.create_client(WaypointSetCurrent, '/mavros/mission/set_current')
        self.waypoint_pull_client = self.create_client(WaypointPull, '/mavros/mission/pull')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.command_client = self.create_client(CommandLong, '/mavros/cmd/command')

        # Wait for services
        self.wait_for_services()

        # Initialize variables
        self.last_status_time = 0
        self.status_interval = 8  # seconds between GCS messages
        self.lasttime = time.time()
        self.run_detection_once = False
        self.waypoint_reached = 0
        
        # Get parameters
        self.get_parameters()
        
        # Setup image handling
        self.bridge = CvBridge()
        self.latest_yolo_image_msg = None
        self.latest_image_msg = None
        
        # Setup directories
        self.setup_directories()
        
        # Initialize geofence
        self.setup_geofence()
        
        self.within_geofence = False

        self.get_logger().info('YOLO Result Subscriber Node initialized')

    def wait_for_services(self):
        """Wait for required services to become available."""
        services = [
            (self.set_wp_client, '/mavros/mission/set_current'),
            (self.waypoint_pull_client, '/mavros/mission/pull'),
            (self.set_mode_client, '/mavros/set_mode'),
            (self.command_client, '/mavros/cmd/command')
        ]
        
        for client, service_name in services:
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for service {service_name}...')

    def get_parameters(self):
        """Get ROS2 parameters."""
        # Declare parameters with default values
        self.declare_parameter('yolo_class_names', '[]')
        self.declare_parameter('num_waypoints', 0)
        self.declare_parameter('takeoff_index', 0)
        self.declare_parameter('rtl_index', 0)
        self.declare_parameter('next_after_takeoff', 0)
        self.declare_parameter('last_before_rtl', 0)
        
        # Get parameter values
        class_names_str = self.get_parameter('yolo_class_names').value
        self.class_names = eval(class_names_str) if class_names_str != '[]' else []
        
        self.num_waypoints = self.get_parameter('num_waypoints').value
        self.takeoff_index = self.get_parameter('takeoff_index').value
        self.rtl_index = self.get_parameter('rtl_index').value
        self.next_after_takeoff = self.get_parameter('next_after_takeoff').value
        self.last_before_rtl = self.get_parameter('last_before_rtl').value

    def setup_directories(self):
        """Setup directories for saving detected images."""
        try:
            package_share_directory = ament_index_python.get_package_share_directory('drone_object_recognition')
            self.detected_object_path = os.path.join(package_share_directory, "detected")
            
            if not os.path.exists(self.detected_object_path):
                os.makedirs(self.detected_object_path)
        except Exception as e:
            self.get_logger().warn(f"Could not create detected objects directory: {e}")
            self.detected_object_path = "/tmp/detected_objects"
            os.makedirs(self.detected_object_path, exist_ok=True)

    def setup_geofence(self):
        """Setup geofence boundaries."""
        # Runway 1 Geofence (Maryland)
        min_lat1 = min(38.3153622, 38.3156463, 38.3159388, 38.3156653)
        max_lat1 = max(38.3153622, 38.3156463, 38.3159388, 38.3156653)
        min_lon1 = min(-76.5508904, -76.5525976, -76.5525077, -76.5507992)
        max_lon1 = max(-76.5508904, -76.5525976, -76.5525077, -76.5507992)

        # Runway 2 Geofence (Maryland)
        min_lat2 = min(38.3145083, 38.3147814, 38.3144952, 38.3141858)
        max_lat2 = max(38.3145083, 38.3147814, 38.3144952, 38.3141858)
        min_lon2 = min(-76.5458706, -76.5457834, -76.5440708, -76.5441687)
        max_lon2 = max(-76.5458706, -76.5457834, -76.5440708, -76.5441687)

        self.GEOFENCE1 = {
            "min_lat1": min_lat1,
            "max_lat1": max_lat1,
            "min_lon1": min_lon1,
            "max_lon1": max_lon1
        }

        self.GEOFENCE2 = {
            "min_lat2": min_lat2,
            "max_lat2": max_lat2,
            "min_lon2": min_lon2,
            "max_lon2": max_lon2
        }

    def geofence_check(self, msg):
        """Check if drone is within geofence boundaries."""
        lat = msg.latitude
        lon = msg.longitude
        self.within_geofence = (
            (self.GEOFENCE1["min_lat1"] <= lat <= self.GEOFENCE1["max_lat1"] and
             self.GEOFENCE1["min_lon1"] <= lon <= self.GEOFENCE1["max_lon1"]) or
            (self.GEOFENCE2["min_lat2"] <= lat <= self.GEOFENCE2["max_lat2"] and
             self.GEOFENCE2["min_lon2"] <= lon <= self.GEOFENCE2["max_lon2"])
        )

        if self.within_geofence:
            self.get_logger().info(f"{GREEN}Geofence status: Inside{RESET}", throttle_duration_sec=10)
            message = "within geofence"
            self.send_status(message, True)
        else:
            self.get_logger().info(f"{YELLOW}Geofence status: Outside{RESET}", throttle_duration_sec=10)
            message = "NOT within geofence"
            self.send_status(message, True)

    def yolo_image_callback(self, msg):
        """Callback for YOLO image messages."""
        self.latest_yolo_image_msg = msg

    def restart_callback(self, msg):
        """Handle restart commands."""
        if "restart" in msg.text.lower():
            message = "Restarting code"
            self.send_status(message, False)
            self.waypoint_reached = 0
            self.lap = 0
            self.run_detection_once = False
            self.detected_object_waypoints.clear_queue()
            self.get_logger().info(f"Waypoint reached = {self.waypoint_reached}")
            self.get_logger().info(f"Lap = {self.lap}")
            self.get_logger().info(f"Run Detection Once = {self.run_detection_once}")
            q = self.detected_object_waypoints.get_detected_objects()
            self.get_logger().info(f"Queue Size = {len(q)}, Objects in Queue: {list(q)}")

    def trigger_camera(self):
        """Trigger camera capture."""
        self.get_logger().info("Object Detected. Triggering Jetson-side camera")
        msg = Bool()
        msg.data = True
        self.detected_photo_pub.publish(msg)

    def update_waypoint_reached(self, msg):
        """Handle waypoint reached messages."""
        self.waypoint_reached = msg.wp_seq
        q = self.detected_object_waypoints.get_detected_objects()

        if self.waypoint_reached == self.last_before_rtl:
            self.lap += 1

            if len(q) == 0:
                self.get_logger().warn("Queue Empty")
                index = self.rtl_index
                # Note: Waypoint deletion would need to be implemented
                return

            name = q[0]["name"]
            lat = q[0]["latitude"]
            long = q[0]["longitude"]
            index = self.last_before_rtl + 1
            message = f"{name} INSERTED at index: {index}"
            self.send_status(message, False)
            self.change_mode("GUIDED")
            # Note: Waypoint addition would need to be implemented
            self.change_mode("AUTO")

            self.rtl_index = index
            self.last_before_rtl = -1
            return

        if self.waypoint_reached == self.rtl_index:
            self.get_logger().info(f"{GREEN}Object waypoint reached{RESET}")
            message = "Object waypoint reached. Payload dropping"
            self.send_status(message, False)
            self.change_mode("GUIDED")
            self.change_mode("AUTO")
            self.change_mode("GUIDED")
            self.servo_controller.run_sequence()
            self.get_logger().info("Stopping detection...")
            self.destroy_subscription(self.subscription)

        self.get_logger().info(f"{GREEN}Lap Updated: {self.lap}{RESET}")

    def speed_cb(self, msg):
        """Handle VFR_HUD messages for speed information."""
        self.get_logger().info(f"{BLUE}Current airspeed: {msg.airspeed:.2f}{RESET}", throttle_duration_sec=10)

    def gps_calc(self, gps_lat, gps_lon, target_x, target_y, img_width, img_height, yaw_degrees):
        """
        Calculate GPS coordinates assuming max shift of 100ft (~0.00030 deg) from image center to edge.
        """
        # Max GPS shift from center to edge (in degrees)
        max_deg_shift = 0.00001373  # ~5 feet

        # Compute center of the image
        image_center_x = img_width / 2.0
        image_center_y = img_height / 2.0

        # Pixel displacement from center
        dx_pixels = target_x - image_center_x
        dy_pixels = target_y - image_center_y

        # Max possible pixel distance (diagonal from center to corner)
        max_pixel_distance = math.sqrt((image_center_x) ** 2 + (image_center_y) ** 2)

        # Actual pixel distance from center to target
        actual_pixel_distance = math.sqrt(dx_pixels**2 + dy_pixels**2)

        # Normalize displacement (0 to 1 scale)
        norm_dx = dx_pixels / max_pixel_distance
        norm_dy = dy_pixels / max_pixel_distance

        # Scale normalized values to GPS degree shift (max 0.00030 degrees)
        raw_shift_lon = norm_dx * max_deg_shift
        raw_shift_lat = norm_dy * max_deg_shift

        # Apply yaw rotation (so direction matches drone orientation)
        yaw_rad = math.radians(yaw_degrees)
        rotated_lon = raw_shift_lon * math.cos(yaw_rad) - raw_shift_lat * math.sin(yaw_rad)
        rotated_lat = raw_shift_lon * math.sin(yaw_rad) + raw_shift_lat * math.cos(yaw_rad)

        # Apply shift to original GPS coordinates
        new_gps_lat = gps_lat - rotated_lat
        new_gps_lon = gps_lon + rotated_lon

        return new_gps_lat, new_gps_lon

    def callback(self, msg):
        """Main YOLO detection callback."""
        if msg.detections.detections:
            self.get_logger().info(f"{len(msg.detections.detections)} object(s) detected!")
            
            # Note: GPS data request would need to be implemented
            # For now, using dummy GPS data
            gps_lat = 38.3150000  # Dummy latitude
            gps_lon = -76.5500000  # Dummy longitude
            gps_alt = 20.0  # Dummy altitude
            yaw = 0.0  # Dummy yaw
            
            bbox_coords = msg.detections.detections
            self.get_logger().info(f"Current wp_reached {self.waypoint_reached}", throttle_duration_sec=5)

            if self.within_geofence:
                for i in range(len(bbox_coords)):
                    lat, long = self.gps_calc(
                        gps_lat,
                        gps_lon,
                        bbox_coords[i].bbox.center.position.x,
                        bbox_coords[i].bbox.center.position.y,
                        640,
                        480,
                        yaw,
                    )
                    
                    detected_name = self.class_names[bbox_coords[i].results[0].hypothesis.class_id] if bbox_coords[i].results[0].hypothesis.class_id < len(self.class_names) else "unknown"
                    index = max(self.next_after_takeoff, self.waypoint_reached + 1)
                    
                    if not self.compare_object_names(detected_name):
                        message = f"{detected_name} DETECTED at index {index}"
                        self.send_status(message, False)
                        self.get_logger().info(f"Calculated Position: LAT: {lat}, LONG:{long}")
                        
                        self.detected_object_waypoints.add_object(
                            detected_name,
                            lat,
                            long,
                            ALT,
                            index,
                        )
                        
                        queue_length = len(self.detected_object_waypoints.get_detected_objects())
                        if self.latest_yolo_image_msg is not None and queue_length <= 2:
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            yolo_image = self.bridge.imgmsg_to_cv2(
                                self.latest_yolo_image_msg, desired_encoding="bgr8"
                            )
                            detected_filename = os.path.join(
                                self.detected_object_path,
                                f"detected_photo_{timestamp}.jpg",
                            )
                            cv2.imwrite(detected_filename, yolo_image)
                            self.get_logger().info(f"Photo saved to {detected_filename}")

                        self.get_logger().info(f"Detected objects: {list(self.detected_object_waypoints.get_detected_objects())}")
        else:
            self.get_logger().info("No objects detected.", throttle_duration_sec=5)

    def compare_object_names(self, object_name):
        """Check if object name already exists in queue."""
        q = self.detected_object_waypoints.get_detected_objects()
        return any(item["name"] == object_name for item in q)

    async def set_servo(self, channel, pwm_value):
        """Set servo PWM value."""
        try:
            request = CommandLong.Request()
            request.command = 183  # MAV_CMD_DO_SET_SERVO
            request.param1 = float(channel)
            request.param2 = float(pwm_value)
            
            response = await self.command_client.call_async(request)
            if response.success:
                self.get_logger().info(f"Servo on channel {channel} set to PWM {pwm_value}")
            else:
                self.get_logger().error("Failed to set servo.")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    async def change_mode(self, mode):
        """Change flight mode."""
        self.get_logger().info(f"Setting mode to {mode}...")
        request = SetMode.Request()
        request.custom_mode = mode
        
        try:
            response = await self.set_mode_client.call_async(request)
            if response.mode_sent:
                self.get_logger().info(f"Mode changed to {mode}")
            else:
                self.get_logger().error("Failed to change mode")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def send_status(self, text, throttle):
        """Send status message."""
        now = time.time()

        if throttle and (now - self.last_status_time <= self.status_interval):
            return
            
        msg = StatusText()
        msg.severity = 6  # 6 = NOTICE
        msg.text = text
        self.status_pub.publish(msg)
        self.last_status_time = now

    async def set_mission_index(self, index):
        """Set current mission waypoint index."""
        self.get_logger().info(f"Setting current mission waypoint to {index}")
        request = WaypointSetCurrent.Request()
        request.wp_seq = index
        
        try:
            response = await self.set_wp_client.call_async(request)
            if response.success:
                self.get_logger().info(f"Mission set to waypoint {index}")
            else:
                self.get_logger().error("Failed to set mission index.")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    
    try:
        node = YoloResultSubscriber()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
