#!/bin/bash

# Drone Object Recognition Setup Script
# This script sets up the environment for the drone object recognition system

set -e

echo "Setting up Drone Object Recognition System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if ROS2 Jazzy is installed
check_ros2() {
    print_status "Checking ROS2 Jazzy installation..."
    
    if ! command -v ros2 &> /dev/null; then
        print_error "ROS2 is not installed. Please install ROS2 Jazzy first."
        echo "Visit: https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debians.html"
        exit 1
    fi
    
    # Check if Jazzy is sourced
    if ! ros2 --version | grep -q "jazzy"; then
        print_warning "ROS2 Jazzy might not be sourced. Make sure to source it:"
        echo "source /opt/ros/jazzy/setup.bash"
    fi
    
    print_success "ROS2 Jazzy found"
}

# Install ROS2 dependencies
install_ros2_deps() {
    print_status "Installing ROS2 dependencies..."
    
    sudo apt update
    sudo apt install -y \
        ros-jazzy-cv-bridge \
        ros-jazzy-image-transport \
        ros-jazzy-vision-msgs \
        ros-jazzy-usb-cam \
        python3-pip \
        python3-venv
    
    print_success "ROS2 dependencies installed"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Always use virtual environment to avoid externally-managed-environment error
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        print_status "Installing Python packages..."
        pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    # Create activation script for future use
    cat > activate_env.sh << 'EOF'
#!/bin/bash
# Script to activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "Virtual environment activated. You can now run ROS2 commands."
echo ""
echo "To run the object recognition system:"
echo "ros2 launch drone_object_recognition object_recognition.launch.py"
EOF
    chmod +x activate_env.sh
    print_success "Created activation script: ./activate_env.sh"
}

# Download YOLO model
download_models() {
    print_status "Downloading YOLOv8s model..."
    
    # Activate virtual environment for model download
    source venv/bin/activate
    
    # This will download the model on first run
    python3 -c "
import torch
from ultralytics import YOLO
print('Downloading YOLOv8s model...')
model = YOLO('yolov8s.pt')
print('Model downloaded successfully!')
"
    
    print_success "Models downloaded"
}

# Set up ROS2 workspace
setup_workspace() {
    print_status "Setting up ROS2 workspace..."
    
    # Check if we're in a ROS2 workspace
    if [ ! -f "package.xml" ]; then
        print_error "Not in a ROS2 package directory. Please run this script from the package root."
        exit 1
    fi
    
    # Build the package
    print_status "Building ROS2 package..."
    cd ..
    if [ -f "CMakeLists.txt" ] || [ -f "package.xml" ]; then
        colcon build --packages-select drone_object_recognition
        source install/setup.bash
        print_success "Package built successfully"
    else
        print_warning "Not in a ROS2 workspace. Please build manually:"
        echo "colcon build --packages-select drone_object_recognition"
    fi
}

# Create test script
create_test_script() {
    print_status "Creating test script..."
    
    cat > test_system.sh << 'EOF'
#!/bin/bash

# Test script for drone object recognition system

echo "🧪 Testing Drone Object Recognition System..."

# Check if ROS2 is sourced
if ! command -v ros2 &> /dev/null; then
    echo "ROS2 not found. Please source ROS2:"
    echo "source /opt/ros/jazzy/setup.bash"
    exit 1
fi

# Test 1: Check if package is built
echo "Checking package build..."
if ros2 pkg list | grep -q "drone_object_recognition"; then
    echo "Package found"
else
    echo "Package not found. Please build the package:"
    echo "colcon build --packages-select drone_object_recognition"
    exit 1
fi

# Test 2: Check if nodes can be found
echo "Checking executable nodes..."
if ros2 pkg executables drone_object_recognition | grep -q "yolo_detector.py"; then
    echo "YOLO detector executable found"
else
    echo "YOLO detector executable not found"
fi

if ros2 pkg executables drone_object_recognition | grep -q "mobilenet_classifier.py"; then
    echo "MobileNet classifier executable found"
else
    echo "MobileNet classifier executable not found"
fi

if ros2 pkg executables drone_object_recognition | grep -q "object_recognition_pipeline.py"; then
    echo "Pipeline executable found"
else
    echo "Pipeline executable not found"
fi

# Test 3: Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "
try:
    import torch
    import cv2
    from ultralytics import YOLO
    import tensorflow as tf
    print('All Python dependencies available')
except ImportError as e:
    print(f'Missing dependency: {e}')
    exit(1)
"

echo "System test completed!"
echo ""
echo "To run the system:"
echo "ros2 launch drone_object_recognition object_recognition.launch.py"
EOF

    chmod +x test_system.sh
    print_success "Test script created"
}

# Main setup function
main() {
    echo "Drone Object Recognition Setup"
    echo "================================="
    
    # Check arguments
    USE_VENV=false
    if [ "$1" = "--venv" ]; then
        USE_VENV=true
    fi
    
    # Run setup steps
    check_ros2
    install_ros2_deps
    install_python_deps $1
    download_models
    setup_workspace
    create_test_script
    
    echo ""
    print_success "Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment:"
    echo "   source venv/bin/activate"
    echo "   # OR use the activation script:"
    echo "   ./activate_env.sh"
    echo ""
    echo "2. Test the system:"
    echo "   python3 test_simple.py"
    echo ""
    echo "3. Build ROS2 package:"
    echo "   cd ~/ros2_ws"
    echo "   colcon build --packages-select drone_object_recognition"
    echo "   source install/setup.bash"
    echo ""
    echo "4. Run the recognition system:"
    echo "   ros2 launch drone_object_recognition object_recognition.launch.py"
    echo ""
    echo "Important: Always activate the virtual environment before running ROS2 commands!"
    echo ""
    echo "For more information, see README.md"
}

# Run main function
main "$@"
