#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_name = 'drone_object_recognition'
    pkg_share_dir = get_package_share_directory(pkg_name)
    
    # Launch arguments for maximum quality
    camera_device_arg = DeclareLaunchArgument(
        'camera_device',
        default_value='/dev/video0',
        description='USB camera device path'
    )
    
    camera_width_arg = DeclareLaunchArgument(
        'camera_width',
        default_value='1920',  # Full HD width for maximum quality
        description='Camera image width'
    )
    
    camera_height_arg = DeclareLaunchArgument(
        'camera_height',
        default_value='1080',  # Full HD height for maximum quality
        description='Camera image height'
    )
    
    fps_arg = DeclareLaunchArgument(
        'fps',
        default_value='30',
        description='Camera frame rate'
    )
    
    record_video_arg = DeclareLaunchArgument(
        'record_video',
        default_value='true',
        description='Enable high-quality video recording'
    )
    
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.3',
        description='YOLO confidence threshold'
    )
    
    classification_threshold_arg = DeclareLaunchArgument(
        'classification_threshold',
        default_value='0.15',
        description='MobileNet classification threshold'
    )
    
    # Camera Recorder Node with high-quality settings
    camera_recorder_node = Node(
        package=pkg_name,
        executable='camera_recorder.py',
        name='camera_recorder',
        parameters=[{
            'camera_device': LaunchConfiguration('camera_device'),
            'camera_width': LaunchConfiguration('camera_width'),
            'camera_height': LaunchConfiguration('camera_height'),
            'fps': LaunchConfiguration('fps'),
            'record_video': LaunchConfiguration('record_video')
        }],
        output='screen'
    )
    
    # YOLO Detector Node
    yolo_detector_node = Node(
        package=pkg_name,
        executable='yolo_detector.py',
        name='yolo_detector',
        parameters=[{
            'confidence_threshold': LaunchConfiguration('confidence_threshold')
        }],
        output='screen'
    )
    
    # MobileNet Classifier Node
    mobilenet_classifier_node = Node(
        package=pkg_name,
        executable='mobilenet_classifier.py',
        name='mobilenet_classifier',
        parameters=[{
            'confidence_threshold': LaunchConfiguration('classification_threshold')
        }],
        output='screen'
    )
    
    # Object Recognition Pipeline Node
    object_recognition_pipeline_node = Node(
        package=pkg_name,
        executable='object_recognition_pipeline.py',
        name='object_recognition_pipeline',
        output='screen'
    )
    
    return LaunchDescription([
        camera_device_arg,
        camera_width_arg,
        camera_height_arg,
        fps_arg,
        record_video_arg,
        confidence_threshold_arg,
        classification_threshold_arg,
        camera_recorder_node,
        yolo_detector_node,
        mobilenet_classifier_node,
        object_recognition_pipeline_node
    ])
