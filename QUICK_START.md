# 🚀 Quick Start Guide

## 📁 Clean Project Structure

```
ObjectRec/
├── src/                              # 📝 Source Code
│   ├── live_object_detection.py      # Main unified script
│   ├── live_object_detection_jetson.py # Jetson optimized
│   └── camera_publisher_node.py      # ROS2 camera node
├── docs/                             # 📖 Documentation
│   ├── jetson_setup_guide.md
│   └── README_ROS2_WORKSPACE.md
├── launch/                           # 🚀 ROS2 Launch Files
│   └── object_recognition.launch.py
├── models/                           # 🤖 AI Models
│   ├── yolov8n.pt
│   └── yolov8s.pt
├── tools/                            # 🛠️ Setup Scripts
│   ├── download_models.sh
│   ├── fix_cmake.sh
│   ├── install_cuda_13.sh
│   ├── install_cuda_dependencies.sh
│   └── upgrade_pytorch_for_gpu.sh
├── ros2_ws/                          # 📦 ROS2 Workspace
├── gpuSetup/                         # 🚀 GPU Setup
│   ├── build_pytorch_sm120.sh
│   └── build_pytorch_with_cuda.sh
├── package.xml                       # ROS2 package definition
├── CMakeLists.txt                    # Build configuration
├── requirements.txt                  # Python dependencies
└── README.md                         # Main documentation
```

## 🎯 How to Run

### 1️⃣ Standalone Mode (Simplest)
```bash
python3 src/live_object_detection.py
```

### 2️⃣ Jetson Nano
```bash
python3 src/live_object_detection_jetson.py
```

### 3️⃣ ROS2 Mode
```bash
cd ros2_ws
colcon build --packages-select drone_object_recognition
source install/setup.bash
ros2 launch drone_object_recognition object_recognition.launch.py
```

### 4️⃣ GPU Setup
```bash
./tools/install_cuda_dependencies.sh
./gpuSetup/build_pytorch_with_cuda.sh
```

## ✅ Benefits of This Organization

- 🗂️ **Clean root directory** - Only essential files at top level
- 📁 **Logical grouping** - Related files organized together
- 🧭 **Easy navigation** - Clear folder structure
- 📦 **Self-contained** - Everything in one place
- 🔧 **Easy maintenance** - Clear separation of concerns

## 🎮 Ready to Use!

Your project is now perfectly organized and ready to use! 🎉
