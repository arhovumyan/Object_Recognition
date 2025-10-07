#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch file for real-time phone detection with camera recording.
    This launch file starts the complete system for phone detection.
    """
    
    # Declare launch arguments
    camera_device_arg = DeclareLaunchArgument(
        'camera_device',
        default_value='/dev/video0',
        description='USB camera device path'
    )
    
    camera_width_arg = DeclareLaunchArgument(
        'camera_width',
        default_value='1280',  # Higher resolution for better quality
        description='Camera image width'
    )
    
    camera_height_arg = DeclareLaunchArgument(
        'camera_height',
        default_value='720',   # Higher resolution for better quality
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
        description='Enable video recording'
    )
    
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.3',
        description='YOLO confidence threshold for phone detection'
    )
    
    classification_threshold_arg = DeclareLaunchArgument(
        'classification_threshold',
        default_value='0.15',
        description='MobileNet classification threshold for phones'
    )
    
    # Camera Recorder Node
    camera_recorder_node = Node(
        package='drone_object_recognition',
        executable='camera_recorder.py',
        name='camera_recorder',
        output='screen',
        parameters=[{
            'camera_device': LaunchConfiguration('camera_device'),
            'camera_width': LaunchConfiguration('camera_width'),
            'camera_height': LaunchConfiguration('camera_height'),
            'fps': LaunchConfiguration('fps'),
            'record_video': LaunchConfiguration('record_video')
        }]
    )
    
    # YOLO Detector Node (optimized for phone detection)
    yolo_detector_node = Node(
        package='drone_object_recognition',
        executable='yolo_detector.py',
        name='yolo_detector',
        output='screen',
        parameters=[{
            'confidence_threshold': LaunchConfiguration('confidence_threshold')
        }]
    )
    
    # MobileNet Classifier Node (optimized for phone classification)
    mobilenet_classifier_node = Node(
        package='drone_object_recognition',
        executable='mobilenet_classifier.py',
        name='mobilenet_classifier',
        output='screen',
        parameters=[{
            'classification_threshold': LaunchConfiguration('classification_threshold')
        }]
    )
    
    # Object Recognition Pipeline Node
    pipeline_node = Node(
        package='drone_object_recognition',
        executable='object_recognition_pipeline.py',
        name='object_recognition_pipeline',
        output='screen'
    )
    
    return LaunchDescription([
        # Launch arguments
        camera_device_arg,
        camera_width_arg,
        camera_height_arg,
        fps_arg,
        record_video_arg,
        confidence_threshold_arg,
        classification_threshold_arg,
        
        # Nodes
        camera_recorder_node,
        yolo_detector_node,
        mobilenet_classifier_node,
        pipeline_node
    ])
