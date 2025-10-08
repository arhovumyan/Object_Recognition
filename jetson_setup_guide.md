# Jetson Nano Setup Guide for Object Recognition

## 1. Flash Jetson OS
```bash
# Download JetPack SDK from NVIDIA
# Flash microSD with JetPack 5.1.2 (Ubuntu 20.04) or newer
# This includes CUDA, cuDNN, TensorRT, and Python 3.8+
```

## 2. Install ROS2 Jazzy
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install ROS2 Jazzy
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-jazzy-desktop python3-rosdep python3-colcon-common-extensions
sudo rosdep init
rosdep update

# Install ROS2 vision packages
sudo apt install ros-jazzy-cv-bridge ros-jazzy-image-transport ros-jazzy-vision-msgs
```

## 3. Install Python Dependencies
```bash
# Install pip if not present
sudo apt install python3-pip

# Install PyTorch for Jetson (CUDA enabled)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip3 install ultralytics opencv-python numpy pillow
```

## 4. Test Camera Connection
```bash
# List available cameras
ls /dev/video*

# Test camera with simple capture
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera works!' if cap.isOpened() else 'Camera failed'); cap.release()"
```

## 5. Optimize for Jetson Performance
```bash
# Set maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Verify GPU is working
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"
```

## 6. Run Object Detection
```bash
# Navigate to your project directory
cd /path/to/ObjectRec

# Run the detection script
python3 live_object_detection.py
```
