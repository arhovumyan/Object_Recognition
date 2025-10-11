#!/usr/bin/env python3

"""
ROS2 Launch file for testing YOLO Detection System
This launch file starts the necessary components for testing the YOLO detection script.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
import os


def generate_launch_description():
    # Declare launch arguments
    use_mock_services_arg = DeclareLaunchArgument(
        'use_mock_services',
        default_value='true',
        description='Whether to use mock MAVROS services for testing'
    )
    
    use_camera_arg = DeclareLaunchArgument(
        'use_camera',
        default_value='true',
        description='Whether to start camera publisher for testing'
    )
    
    use_detection_arg = DeclareLaunchArgument(
        'use_detection',
        default_value='true',
        description='Whether to start live object detection'
    )
    
    # Mock MAVROS Services Node (for testing)
    mock_mavros_node = Node(
        package='drone_object_recognition',
        executable='mock_mavros_services.py',
        name='mock_mavros_services',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_mock_services')),
        parameters=[{
            'mock_gps_lat': 38.3150000,
            'mock_gps_lon': -76.5500000,
            'mock_gps_alt': 20.0,
            'mock_yaw': 0.0,
            'mock_waypoint_reached': 0
        }]
    )
    
    # Camera Publisher Node
    camera_publisher_node = Node(
        package='drone_object_recognition',
        executable='camera_publisher_node.py',
        name='camera_publisher',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_camera'))
    )
    
    # Live Object Detection Node
    object_detection_node = Node(
        package='drone_object_recognition',
        executable='live_object_detection.py',
        name='object_detection',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_detection')),
        arguments=['--ros-args'],
        parameters=[{
            'camera_topic': '/camera/image_raw',
            'confidence_threshold': 0.3,
            'use_classifier': True,
            'classifier_threshold': 0.6
        }]
    )
    
    # YOLO Detection ROS2 Node
    yolo_detection_node = Node(
        package='drone_object_recognition',
        executable='yolo_detection_ros2.py',
        name='yolo_result_subscriber',
        output='screen',
        parameters=[{
            'yolo_class_names': "['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']",
            'num_waypoints': 10,
            'takeoff_index': 0,
            'rtl_index': 9,
            'next_after_takeoff': 1,
            'last_before_rtl': 8
        }]
    )
    
    # Test Data Publisher Node (publishes mock YOLO detection results)
    test_data_publisher_node = Node(
        package='drone_object_recognition',
        executable='test_yolo_publisher.py',
        name='test_yolo_publisher',
        output='screen',
        parameters=[{
            'publish_rate': 1.0,  # Publish every 1 second
            'detect_person': True,
            'detect_car': False,
            'detect_bottle': True
        }]
    )
    
    return LaunchDescription([
        use_mock_services_arg,
        use_camera_arg,
        use_detection_arg,
        mock_mavros_node,
        camera_publisher_node,
        object_detection_node,
        yolo_detection_node,
        test_data_publisher_node
    ])
