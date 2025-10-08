#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch file for the complete object recognition system (CPU-only mode).
    This version forces CPU usage for all models to avoid GPU compatibility issues.
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
    
    classification_threshold_arg = DeclareLaunchArgument(
        'classification_threshold',
        default_value='0.3',
        description='MobileNet classification threshold'
    )
    
    # Environment variable to force CPU usage
    cpu_env = {'CUDA_VISIBLE_DEVICES': ''}
    
    # YOLO Detector Node (CPU-only)
    yolo_detector_node = Node(
        package='drone_object_recognition',
        executable='scripts/yolo_detector_wrapper.py',
        name='yolo_detector',
        output='screen',
        parameters=[{
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'camera_topic': LaunchConfiguration('camera_topic')
        }],
        remappings=[
            ('/camera/image_raw', LaunchConfiguration('camera_topic'))
        ],
        env=cpu_env
    )
    
    # MobileNet Classifier Node (CPU-only)
    mobilenet_classifier_node = Node(
        package='drone_object_recognition',
        executable='scripts/mobilenet_classifier_wrapper.py',
        name='mobilenet_classifier',
        output='screen',
        parameters=[{
            'classification_threshold': LaunchConfiguration('classification_threshold')
        }],
        env=cpu_env
    )
    
    # Object Recognition Pipeline Node
    pipeline_node = Node(
        package='drone_object_recognition',
        executable='scripts/object_recognition_pipeline_wrapper.py',
        name='object_recognition_pipeline',
        output='screen',
        env=cpu_env
    )
    
    return LaunchDescription([
        camera_topic_arg,
        confidence_threshold_arg,
        classification_threshold_arg,
        yolo_detector_node,
        mobilenet_classifier_node,
        pipeline_node
    ])
