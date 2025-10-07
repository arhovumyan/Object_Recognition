#!/bin/bash
# Script to activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "Virtual environment activated. You can now run ROS2 commands."
echo ""
echo "To test the system:"
echo "python3 test_simple.py"
echo ""
echo "To run the object recognition system:"
echo "ros2 launch drone_object_recognition object_recognition.launch.py"
