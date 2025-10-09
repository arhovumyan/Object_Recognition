# 🚁 Drone Object Recognition System

A unified object detection system that works both as standalone scripts and as ROS2 nodes. Optimized for real-time performance with GPU acceleration support.

## ✨ Features

- **🤖 YOLOv8 Object Detection** - Fast and accurate object detection
- **🔄 Unified System** - Works standalone and as ROS2 node
- **🚀 GPU Acceleration** - CUDA support for high performance
- **📱 Multi-Platform** - Desktop and Jetson Nano support
- **📡 ROS2 Integration** - Full ROS2 ecosystem compatibility
- **⚡ Real-time Processing** - Optimized for live video streams

## 🎯 Quick Start

### Standalone Mode (No ROS2 needed)
```bash
# Main detection script
python3 src/live_object_detection.py

# Jetson Nano optimized
python3 src/live_object_detection_jetson.py
```

### ROS2 Mode
```bash
cd ros2_ws
colcon build --packages-select drone_object_recognition
source install/setup.bash
ros2 launch drone_object_recognition object_recognition.launch.py
```

## 📁 Project Structure

```
ObjectRec/
├── src/                    # Source code
├── docs/                   # Documentation
├── launch/                 # ROS2 launch files
├── models/                 # AI model files
├── tools/                  # Setup scripts
├── ros2_ws/               # ROS2 workspace
└── gpuSetup/              # GPU setup
```

## 🚀 GPU Setup

For GPU acceleration:
```bash
./tools/install_cuda_dependencies.sh
./gpuSetup/build_pytorch_with_cuda.sh
```

## 📖 Documentation

- [Jetson Nano Setup](docs/jetson_setup_guide.md)
- [ROS2 Workspace Guide](docs/README_ROS2_WORKSPACE.md)

## 🎮 Usage Examples

| Mode | Command | Description |
|------|---------|-------------|
| **Standalone** | `python3 src/live_object_detection.py` | Direct camera access |
| **Jetson** | `python3 src/live_object_detection_jetson.py` | Jetson optimized |
| **ROS2** | `ros2 launch drone_object_recognition object_recognition.launch.py` | Full ROS2 system |

## 📊 Performance

- **GPU Mode**: 60-120 FPS (RTX 5070)
- **CPU Mode**: 10-20 FPS
- **Jetson Nano**: 15-30 FPS with GPU

## 🔧 Requirements

- Python 3.10+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- ROS2 Jazzy (for ROS2 mode)

## 📄 License

MIT License
