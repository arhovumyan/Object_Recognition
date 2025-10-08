#!/bin/bash

# CPU-only test script for object recognition system
# This script runs the system in CPU-only mode to avoid GPU compatibility issues

echo "Drone Object Recognition System - CPU-Only Test"
echo "=============================================="
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "No virtual environment found. Using system Python..."
fi

echo "Running CPU-only component tests..."
echo ""

# Run the CPU-only test
python3 test_cpu_only.py

echo ""
echo "Test completed!"
echo ""
echo "If all tests passed, you can now run the system with:"
echo "  ros2 launch drone_object_recognition object_recognition_cpu.launch.py"
echo ""
echo "Or test with USB camera:"
echo "  ros2 launch drone_object_recognition test_with_usb_camera_cpu.launch.py"
