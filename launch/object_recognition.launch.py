#!/usr/bin/env python3

"""
ROS2 Launch file for Object Recognition System
Launches camera publisher and object detection nodes.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition


def generate_launch_description():
    # Declare launch arguments
    use_camera_node_arg = DeclareLaunchArgument(
        'use_camera_node',
        default_value='true',
        description='Whether to use the camera publisher node (true) or external camera topic (false)'
    )
    
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Camera topic to subscribe to'
    )
    
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.3',
        description='Detection confidence threshold'
    )
    
    # Camera Publisher Node (optional)
    camera_publisher_node = Node(
        package='drone_object_recognition',
        executable='camera_publisher_node.py',
        name='camera_publisher',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_camera_node'))
    )
    
    # Object Detection Node (Unified)
    object_detection_node = Node(
        package='drone_object_recognition',
        executable='live_object_detection.py',
        name='object_detection',
        output='screen',
        arguments=['--ros-args'],
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold')
        }]
    )
    
    return LaunchDescription([
        use_camera_node_arg,
        camera_topic_arg,
        confidence_threshold_arg,
        camera_publisher_node,
        object_detection_node
    ])
