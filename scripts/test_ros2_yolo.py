#!/usr/bin/env python3

"""
Simple Test Script for ROS2 YOLO Detection System
This script provides easy commands to test different components of the system.
"""

import subprocess
import time
import sys
import os


def run_command(cmd, description):
    """Run a command and print the description."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, timeout=10)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out after 10 seconds (this is normal for long-running processes)")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_individual_components():
    """Test individual components of the system."""
    print("🧪 Testing Individual Components")
    
    # Test 1: Mock MAVROS Services
    run_command(
        "cd /home/aro/Documents/ObjectRec/ros2_ws && bash -c 'source install/setup.bash && timeout 5s ros2 run drone_object_recognition mock_mavros_services.py'",
        "Testing Mock MAVROS Services"
    )
    
    # Test 2: Test YOLO Publisher
    run_command(
        "cd /home/aro/Documents/ObjectRec/ros2_ws && bash -c 'source install/setup.bash && timeout 5s ros2 run drone_object_recognition test_yolo_publisher.py'",
        "Testing YOLO Data Publisher"
    )
    
    # Test 3: YOLO Detection Node
    run_command(
        "cd /home/aro/Documents/ObjectRec/ros2_ws && bash -c 'source install/setup.bash && timeout 5s ros2 run drone_object_recognition yolo_detection_ros2.py'",
        "Testing YOLO Detection Node"
    )


def test_integration():
    """Test the integrated system."""
    print("\n🔗 Testing Integrated System")
    
    # Test with launch file
    run_command(
        "cd /home/aro/Documents/ObjectRec/ros2_ws && bash -c 'source install/setup.bash && timeout 10s ros2 launch drone_object_recognition test_yolo_detection.launch.py'",
        "Testing Full System Integration"
    )


def test_topics():
    """Test ROS2 topics."""
    print("\n📡 Testing ROS2 Topics")
    
    run_command(
        "cd /home/aro/Documents/ObjectRec/ros2_ws && bash -c 'source install/setup.bash && ros2 topic list'",
        "Listing Available Topics"
    )
    
    run_command(
        "cd /home/aro/Documents/ObjectRec/ros2_ws && bash -c 'source install/setup.bash && ros2 topic info /yolo_result'",
        "YOLO Result Topic Info"
    )


def test_services():
    """Test ROS2 services."""
    print("\n🔧 Testing ROS2 Services")
    
    run_command(
        "cd /home/aro/Documents/ObjectRec/ros2_ws && bash -c 'source install/setup.bash && ros2 service list'",
        "Listing Available Services"
    )


def interactive_test():
    """Interactive testing mode."""
    print("\n🎮 Interactive Testing Mode")
    print("Choose what to test:")
    print("1. Individual Components")
    print("2. Integration Test")
    print("3. Topics Test")
    print("4. Services Test")
    print("5. Run Full System (with camera)")
    print("6. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            test_individual_components()
        elif choice == "2":
            test_integration()
        elif choice == "3":
            test_topics()
        elif choice == "4":
            test_services()
        elif choice == "5":
            print("\n🚀 Starting Full System Test...")
            print("This will start all components including camera.")
            print("Press Ctrl+C to stop.")
            run_command(
                "cd /home/aro/Documents/ObjectRec/ros2_ws && bash -c 'source install/setup.bash && ros2 launch drone_object_recognition test_yolo_detection.launch.py'",
                "Full System Test"
            )
        elif choice == "6":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-6.")


def main():
    """Main function."""
    print("🤖 ROS2 YOLO Detection System Test Suite")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "components":
            test_individual_components()
        elif test_type == "integration":
            test_integration()
        elif test_type == "topics":
            test_topics()
        elif test_type == "services":
            test_services()
        elif test_type == "full":
            print("🚀 Starting Full System Test...")
            run_command(
                "cd /home/aro/Documents/ObjectRec/ros2_ws && bash -c 'source install/setup.bash && ros2 launch drone_object_recognition test_yolo_detection.launch.py'",
                "Full System Test"
            )
        else:
            print("❌ Unknown test type. Use: components, integration, topics, services, or full")
    else:
        interactive_test()


if __name__ == "__main__":
    main()
