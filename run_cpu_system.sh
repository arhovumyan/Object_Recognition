#!/bin/bash

# CPU-only object recognition system launcher
# This script runs the complete object recognition system in CPU-only mode

echo "Drone Object Recognition System - CPU-Only Mode"
echo "=============================================="
echo ""

# Check if virtual environment exists and activate it
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "No virtual environment found. Using system Python..."
fi

# Source ROS2 setup
echo "Sourcing ROS2 setup..."
source /opt/ros/jazzy/setup.bash

# Source the workspace
echo "Sourcing workspace..."
source install/setup.bash

echo ""
echo "Starting CPU-only object recognition system..."
echo "This will run YOLO and MobileNet models on CPU only."
echo ""
echo "Press Ctrl+C to stop the system."
echo ""

# Run the CPU-only launch file
ros2 launch drone_object_recognition object_recognition_cpu.launch.py
