#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch file for YOLO detection only (without MobileNet classification).
    Useful for testing or when only detection is needed.
    """
    
    # Declare launch arguments
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Camera topic to subscribe to'
    )
    
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='YOLO confidence threshold'
    )
    
    # YOLO Detector Node
    yolo_detector_node = Node(
        package='drone_object_recognition',
        executable='yolo_detector.py',
        name='yolo_detector',
        output='screen',
        parameters=[{
            'confidence_threshold': LaunchConfiguration('confidence_threshold')
        }],
        remappings=[
            ('/camera/image_raw', LaunchConfiguration('camera_topic'))
        ]
    )
    
    return LaunchDescription([
        camera_topic_arg,
        confidence_threshold_arg,
        yolo_detector_node
    ])
