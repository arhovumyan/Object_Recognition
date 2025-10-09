#!/bin/bash

echo "🔧 Fixing CMake Configuration Issues..."
echo "====================================="
echo "📍 ROS2 Workspace: /home/aro/Documents/ObjectRec/ros2_ws"

# Clean build directory
echo "🧹 Cleaning build directory..."
cd /home/aro/Documents/ObjectRec/ros2_ws
rm -rf build/drone_object_recognition
rm -rf install/drone_object_recognition
rm -rf log/latest_build/drone_object_recognition

# Update source files
echo "📁 Updating source files..."
cp -r /home/aro/Documents/ObjectRec/* /home/aro/Documents/ObjectRec/ros2_ws/src/drone_object_recognition/

# Rebuild package
echo "🔨 Rebuilding package..."
colcon build --packages-select drone_object_recognition

# Source workspace
echo "📦 Sourcing workspace..."
source install/setup.bash

echo "✅ CMake issues should be resolved!"
echo ""
echo "🚀 To test the system:"
echo "   ros2 launch drone_object_recognition object_recognition.launch.py"
