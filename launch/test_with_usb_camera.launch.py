#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch file for testing with USB camera.
    Includes camera node and object recognition system.
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
            'image_width': 640,
            'image_height': 480,
            'camera_info_url': ''
        }],
        remappings=[
            ('/image_raw', LaunchConfiguration('camera_topic'))
        ]
    )
    
    # YOLO Detector Node
    yolo_detector_node = Node(
        package='drone_object_recognition',
        executable='yolo_detector.py',
        name='yolo_detector',
        output='screen',
        remappings=[
            ('/camera/image_raw', LaunchConfiguration('camera_topic'))
        ]
    )
    
    # MobileNet Classifier Node
    mobilenet_classifier_node = Node(
        package='drone_object_recognition',
        executable='mobilenet_classifier.py',
        name='mobilenet_classifier',
        output='screen'
    )
    
    # Object Recognition Pipeline Node
    pipeline_node = Node(
        package='drone_object_recognition',
        executable='object_recognition_pipeline.py',
        name='object_recognition_pipeline',
        output='screen'
    )
    
    return LaunchDescription([
        camera_device_arg,
        camera_topic_arg,
        usb_camera_node,
        yolo_detector_node,
        mobilenet_classifier_node,
        pipeline_node
    ])
