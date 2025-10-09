# 🔧 Troubleshooting Guide

## ❌ CUDA Error: "no kernel image is available for execution on the device"

This error occurs when PyTorch was compiled for a different CUDA architecture than your GPU supports.

### 🔍 Diagnosis
The error means:
- Your GPU exists and CUDA is available
- But PyTorch was compiled with CUDA kernels that don't match your GPU's compute capability
- This commonly happens with older GPUs or mismatched PyTorch installations

### 🚀 Quick Fixes

#### 1️⃣ **Use CPU-Only Mode (Immediate Solution)**
```bash
# Run the CPU-only version
python3 src/live_object_detection_cpu.py
```

#### 2️⃣ **Force CPU in Environment**
```bash
export CUDA_VISIBLE_DEVICES=''
python3 src/live_object_detection.py
```

#### 3️⃣ **Reinstall PyTorch with Correct CUDA Version**
```bash
# Check your CUDA version
nvidia-smi

# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Install PyTorch with CUDA 12.1 (most compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or install CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 4️⃣ **For ROS2, Use CPU Mode**
```bash
# Modify the launch file to force CPU mode
export CUDA_VISIBLE_DEVICES=''
ros2 launch drone_object_recognition object_recognition.launch.py
```

### 🧪 Test Your System

Run the diagnostic script:
```bash
python3 test_system.py
```

This will test:
- ✅ Package imports
- ✅ Camera access
- ✅ YOLO CPU inference
- ✅ ROS2 basic functionality

### 📊 Expected Performance

| Mode | Expected FPS | Use Case |
|------|--------------|----------|
| **GPU (Fixed)** | 60-120 FPS | High performance |
| **CPU** | 10-20 FPS | Development/testing |
| **Jetson GPU** | 15-30 FPS | Embedded systems |

### 🎯 Working Commands

#### Standalone Mode (CPU):
```bash
python3 src/live_object_detection_cpu.py
```

#### ROS2 Mode (CPU):
```bash
export CUDA_VISIBLE_DEVICES=''
cd ros2_ws
source install/setup.bash
ros2 launch drone_object_recognition object_recognition.launch.py
```

#### Jetson Nano:
```bash
python3 src/live_object_detection_jetson.py
```

### 🔍 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **CUDA kernel error** | Use CPU mode or reinstall PyTorch |
| **Camera not found** | Check permissions: `sudo usermod -a -G video $USER` |
| **ROS2 not working** | Source workspace: `source install/setup.bash` |
| **Low FPS** | Reduce resolution or use GPU acceleration |
| **Import errors** | Install requirements: `pip install -r requirements.txt` |

### 🚀 Performance Optimization

#### For CPU Mode:
- Use `yolov8n.pt` (nano model)
- Reduce input resolution to 320x320
- Skip frames (process every 2nd frame)
- Lower confidence threshold

#### For GPU Mode:
- Use `yolov8s.pt` (small model)
- Increase input resolution to 640x640
- Process every frame
- Higher confidence threshold

### 📞 Still Having Issues?

1. **Run diagnostics**: `python3 test_system.py`
2. **Check logs**: Look at the error messages carefully
3. **Try CPU mode first**: `python3 src/live_object_detection_cpu.py`
4. **Verify dependencies**: `pip install -r requirements.txt`

The system is designed to be robust and should work in CPU mode even if GPU acceleration fails.
