#!/bin/bash

# Fix script for externally-managed-environment error
# This script creates a virtual environment and installs dependencies

set -e

echo "🔧 Fixing Python environment..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Create virtual environment
print_status "Creating virtual environment..."
python3 -m venv venv
print_success "Virtual environment created"

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_status "Installing Python dependencies..."
pip install -r requirements.txt
print_success "Python dependencies installed"

# Download YOLO model
print_status "Downloading YOLOv8s model..."
python3 -c "
import torch
from ultralytics import YOLO
print('Downloading YOLOv8s model...')
model = YOLO('yolov8s.pt')
print('Model downloaded successfully!')
"
print_success "YOLO model downloaded"

# Create activation script
cat > activate_env.sh << 'EOF'
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
EOF
chmod +x activate_env.sh

print_success "Environment fixed successfully! 🎉"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo "   # OR use: ./activate_env.sh"
echo ""
echo "2. Test the system:"
echo "   python3 test_simple.py"
echo ""
echo "3. Build ROS2 package:"
echo "   cd ~/ros2_ws"
echo "   colcon build --packages-select drone_object_recognition"
echo "   source install/setup.bash"
echo ""
echo "4. Run the system:"
echo "   ros2 launch drone_object_recognition object_recognition.launch.py"
echo ""
echo "Important: Always activate the virtual environment before running commands!"
