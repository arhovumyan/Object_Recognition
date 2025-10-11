#!/usr/bin/env python3

"""
Mock MAVROS Services for Testing YOLO Detection System
This script provides mock MAVROS services so you can test the YOLO detection without a real drone.
"""

import rclpy
from rclpy.node import Node
from mavros_msgs.srv import (
    CommandLong,
    SetMode,
    WaypointSetCurrent,
    WaypointPull,
)
from mavros_msgs.msg import WaypointReached, StatusText
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool
import time
import math


class MockMavrosServices(Node):
    """Mock MAVROS services for testing."""
    
    def __init__(self):
        super().__init__('mock_mavros_services')
        
        # Get parameters
        self.declare_parameter('mock_gps_lat', 38.3150000)
        self.declare_parameter('mock_gps_lon', -76.5500000)
        self.declare_parameter('mock_gps_alt', 20.0)
        self.declare_parameter('mock_yaw', 0.0)
        self.declare_parameter('mock_waypoint_reached', 0)
        
        self.mock_gps_lat = self.get_parameter('mock_gps_lat').value
        self.mock_gps_lon = self.get_parameter('mock_gps_lon').value
        self.mock_gps_alt = self.get_parameter('mock_gps_alt').value
        self.mock_yaw = self.get_parameter('mock_yaw').value
        self.mock_waypoint_reached = self.get_parameter('mock_waypoint_reached').value
        
        # Service servers
        self.set_wp_service = self.create_service(
            WaypointSetCurrent,
            '/mavros/mission/set_current',
            self.set_current_waypoint_callback
        )
        
        self.waypoint_pull_service = self.create_service(
            WaypointPull,
            '/mavros/mission/pull',
            self.waypoint_pull_callback
        )
        
        self.set_mode_service = self.create_service(
            SetMode,
            '/mavros/set_mode',
            self.set_mode_callback
        )
        
        self.command_service = self.create_service(
            CommandLong,
            '/mavros/cmd/command',
            self.command_callback
        )
        
        # Publishers
        self.waypoint_reached_pub = self.create_publisher(
            WaypointReached,
            '/mavros/mission/reached',
            10
        )
        
        self.gps_pub = self.create_publisher(
            NavSatFix,
            '/mavros/global_position/global',
            10
        )
        
        self.status_pub = self.create_publisher(
            StatusText,
            '/mavros/statustext/send',
            10
        )
        
        # Timer for publishing mock data
        self.timer = self.create_timer(1.0, self.publish_mock_data)
        
        self.get_logger().info('Mock MAVROS Services started')
        self.get_logger().info(f'Mock GPS: LAT={self.mock_gps_lat}, LON={self.mock_gps_lon}, ALT={self.mock_gps_alt}')
    
    def set_current_waypoint_callback(self, request, response):
        """Handle waypoint set current requests."""
        self.get_logger().info(f'Setting current waypoint to: {request.wp_seq}')
        self.mock_waypoint_reached = request.wp_seq
        response.success = True
        return response
    
    def waypoint_pull_callback(self, request, response):
        """Handle waypoint pull requests."""
        self.get_logger().info('Waypoint pull requested')
        response.success = True
        return response
    
    def set_mode_callback(self, request, response):
        """Handle set mode requests."""
        self.get_logger().info(f'Setting mode to: {request.custom_mode}')
        response.mode_sent = True
        return response
    
    def command_callback(self, request, response):
        """Handle command requests (like servo control)."""
        self.get_logger().info(f'Command received: {request.command}, params: {request.param1}, {request.param2}')
        response.success = True
        return response
    
    def publish_mock_data(self):
        """Publish mock GPS and waypoint data."""
        # Publish mock GPS data
        gps_msg = NavSatFix()
        gps_msg.header.stamp = self.get_clock().now().to_msg()
        gps_msg.header.frame_id = "global"
        gps_msg.latitude = self.mock_gps_lat
        gps_msg.longitude = self.mock_gps_lon
        gps_msg.altitude = self.mock_gps_alt
        gps_msg.status.status = 0  # STATUS_FIX
        gps_msg.status.service = 1  # SERVICE_GPS
        self.gps_pub.publish(gps_msg)
        
        # Occasionally publish waypoint reached (simulate mission progress)
        if time.time() % 10 < 1:  # Every ~10 seconds
            wp_msg = WaypointReached()
            wp_msg.header.stamp = self.get_clock().now().to_msg()
            wp_msg.header.frame_id = "base_link"
            wp_msg.wp_seq = self.mock_waypoint_reached
            self.waypoint_reached_pub.publish(wp_msg)
            self.get_logger().info(f'Published waypoint reached: {self.mock_waypoint_reached}')
    
    def simulate_object_detection_trigger(self):
        """Simulate triggering object detection by advancing waypoint."""
        self.mock_waypoint_reached += 1
        self.get_logger().info(f'Simulated object detection - advancing to waypoint: {self.mock_waypoint_reached}')


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    
    try:
        node = MockMavrosServices()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
