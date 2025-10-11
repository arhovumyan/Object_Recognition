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
    
    # Two-stage pipeline parameters
    use_classifier_arg = DeclareLaunchArgument(
        'use_classifier',
        default_value='true',
        description='Enable MobileNetV3 classifier for two-stage detection'
    )
    
    classifier_threshold_arg = DeclareLaunchArgument(
        'classifier_threshold',
        default_value='0.3',
        description='MobileNetV3 classification confidence threshold'
    )
    
    classifier_model_path_arg = DeclareLaunchArgument(
        'classifier_model_path',
        default_value='models/mobilenetv3_tent_mannequin.pth',
        description='Path to fine-tuned MobileNetV3 model (optional)'
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
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'use_classifier': LaunchConfiguration('use_classifier'),
            'classifier_threshold': LaunchConfiguration('classifier_threshold'),
            'classifier_model_path': LaunchConfiguration('classifier_model_path')
        }]
    )
    
    return LaunchDescription([
        use_camera_node_arg,
        camera_topic_arg,
        confidence_threshold_arg,
        use_classifier_arg,
        classifier_threshold_arg,
        classifier_model_path_arg,
        camera_publisher_node,
        object_detection_node
    ])
