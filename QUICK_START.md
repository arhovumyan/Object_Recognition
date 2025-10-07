# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Setup Environment
```bash
# Navigate to your package directory
cd /home/aro/Documents/ObjectRec

# Run the setup script
./setup.sh

# Or use virtual environment (recommended)
./setup.sh --venv
```

### 2. Test Components
```bash
# Test all components without ROS2
python3 test_simple.py
```

### 3. Build ROS2 Package
```bash
# Navigate to ROS2 workspace
cd ~/ros2_ws  # or your workspace directory

# Copy package to src
cp -r /home/aro/Documents/ObjectRec/drone_object_recognition src/

# Build package
colcon build --packages-select drone_object_recognition
source install/setup.bash
```

### 4. Run the System

#### With USB Camera (for testing)
```bash
ros2 launch drone_object_recognition test_with_usb_camera.launch.py
```

#### With existing camera topic
```bash
ros2 launch drone_object_recognition object_recognition.launch.py \
    camera_topic:=/your_camera_topic
```

#### YOLO only (detection without classification)
```bash
ros2 launch drone_object_recognition yolo_only.launch.py
```

### 5. Monitor Results
```bash
# Check system status
ros2 topic echo /recognition_status

# Monitor detections
ros2 topic echo /object_detections

# Monitor classifications
ros2 topic echo /object_classification

# Check if target found
ros2 topic echo /target_object_found
```

### 6. Change Target Object
```bash
# Available targets: toothbrush, mouse, phone, hat
ros2 topic pub /recognition_command std_msgs/String "data: 'phone'"
```

## 📊 Expected Performance

- **YOLOv8s**: ~30-60 FPS on GPU, ~5-15 FPS on CPU
- **MobileNetV3**: ~100+ FPS on GPU, ~20-50 FPS on CPU
- **Total System**: ~5-30 FPS depending on hardware

## 🎯 Target Objects

The system detects these objects:
- **Phone**: Detected as "cell phone" by YOLO
- **Mouse**: Detected as "cell phone" by YOLO (closest match)
- **Hat**: Detected as "person" by YOLO, classified as hat-wearing
- **Toothbrush**: Not in YOLO dataset (requires custom training)

## 🔧 Troubleshooting

### Common Issues

1. **"No module named 'ultralytics'"**
   ```bash
   pip install ultralytics
   ```

2. **"No module named 'tensorflow'"**
   ```bash
   pip install tensorflow
   ```

3. **Camera not found**
   ```bash
   # Check available cameras
   ls /dev/video*
   
   # Test with different device
   ros2 launch drone_object_recognition test_with_usb_camera.launch.py camera_device:=/dev/video1
   ```

4. **Low FPS**
   ```bash
   # Lower confidence thresholds
   ros2 launch drone_object_recognition object_recognition.launch.py confidence_threshold:=0.3
   ```

### Debug Commands
```bash
# Check if nodes are running
ros2 node list

# Check node info
ros2 node info /yolo_detector
ros2 node info /mobilenet_classifier

# Monitor debug images
rqt_image_view
```

## 📱 Integration with Drone

The system publishes target positions for drone control:
```bash
# Monitor target positions
ros2 topic echo /target_position
```

Use the position data for drone navigation:
- `x, y`: Image coordinates of target center
- `theta`: Detection confidence (0.0 to 1.0)

## 🎮 Manual Testing

1. Point camera at objects
2. Check `/target_object_found` topic
3. Verify debug images show bounding boxes
4. Test different target objects via commands

## 📈 Performance Tips

1. **Use GPU**: Ensure CUDA is installed for faster inference
2. **Lower Resolution**: Reduce camera resolution for higher FPS
3. **Adjust Thresholds**: Lower confidence thresholds for more detections
4. **Single Thread**: Run on single CPU core for consistent timing

## 🔄 Next Steps

1. **Custom Training**: Train YOLO on your specific objects
2. **Integration**: Connect to your drone's flight controller
3. **Optimization**: Use TensorRT for Jetson devices
4. **Customization**: Modify target objects and thresholds
