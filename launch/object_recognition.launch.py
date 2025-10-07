#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch file for the complete object recognition system.
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
    
    # YOLO Detector Node
    yolo_detector_node = Node(
        package='drone_object_recognition',
        executable='yolo_detector.py',
        name='yolo_detector',
        output='screen',
        parameters=[{
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'camera_topic': LaunchConfiguration('camera_topic')
        }],
        remappings=[
            ('/camera/image_raw', LaunchConfiguration('camera_topic'))
        ]
    )
    
    # MobileNet Classifier Node
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
        camera_topic_arg,
        confidence_threshold_arg,
        classification_threshold_arg,
        yolo_detector_node,
        mobilenet_classifier_node,
        pipeline_node
    ])
