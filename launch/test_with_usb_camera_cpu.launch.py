#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch file for testing with USB camera (CPU-only mode).
    Includes camera node and object recognition system with forced CPU usage.
    """
    
    # Declare launch arguments
    camera_device_arg = DeclareLaunchArgument(
        'camera_device',
        default_value='/dev/video0',
        description='USB camera device path'
    )
    
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Camera topic to publish to'
    )
    
    # Environment variable to force CPU usage
    cpu_env = {'CUDA_VISIBLE_DEVICES': ''}
    
    # USB Camera Node (using usb_cam package)
    usb_camera_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_camera',
        output='screen',
        parameters=[{
            'video_device': LaunchConfiguration('camera_device'),
            'framerate': 30.0,
            'pixel_format': 'yuyv',
            'image_width': 1280,
            'image_height': 720,
            'camera_info_url': ''
        }],
        remappings=[
            ('/image_raw', LaunchConfiguration('camera_topic'))
        ]
    )
    
    # YOLO Detector Node (CPU-only)
    yolo_detector_node = Node(
        package='drone_object_recognition',
        executable='scripts/yolo_detector_wrapper.py',
        name='yolo_detector',
        output='screen',
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'confidence_threshold': 0.25,
            'phone_confidence_threshold': 0.12,
            'iou_threshold': 0.45,
            'imgsz': 1280
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
        camera_device_arg,
        camera_topic_arg,
        usb_camera_node,
        yolo_detector_node,
        mobilenet_classifier_node,
        pipeline_node
    ])
