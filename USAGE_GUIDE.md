# 🚁 Drone Object Recognition - Usage Guide

## ✅ System Status: READY TO USE!

Your drone object recognition system is now fully set up and tested. All components are working correctly.

## 🚀 Quick Start Commands

### 1. Activate Environment
```bash
cd /home/aro/Documents/ObjectRec
source venv/bin/activate
# OR use the convenience script:
./activate_env.sh
```

### 2. Run the System

#### Option A: Full System (YOLO + MobileNet)
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch drone_object_recognition object_recognition.launch.py
```

#### Option B: YOLO Detection Only
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch drone_object_recognition yolo_only.launch.py
```

#### Option C: With USB Camera (for testing)
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch drone_object_recognition test_with_usb_camera.launch.py
```

## 📊 Monitor Results

### Check System Status
```bash
ros2 topic echo /recognition_status
```

### Monitor Detections
```bash
ros2 topic echo /object_detections
```

### Monitor Classifications
```bash
ros2 topic echo /object_classification
```

### Check if Target Found
```bash
ros2 topic echo /target_object_found
```

### View Debug Images
```bash
rqt_image_view
# Then select topics: /yolo_debug_image or /classification_debug_image
```

## 🎯 Change Target Object
```bash
# Available targets: toothbrush, mouse, phone, hat
ros2 topic pub /recognition_command std_msgs/String "data: 'phone'"
ros2 topic pub /recognition_command std_msgs/String "data: 'mouse'"
ros2 topic pub /recognition_command std_msgs/String "data: 'hat'"
```

## 🔧 Custom Parameters

### Adjust Confidence Thresholds
```bash
ros2 launch drone_object_recognition object_recognition.launch.py \
    confidence_threshold:=0.3 \
    classification_threshold:=0.4
```

### Use Different Camera Topic
```bash
ros2 launch drone_object_recognition object_recognition.launch.py \
    camera_topic:=/your_camera/image_raw
```

## 📈 Expected Performance

- **YOLOv8s**: ~30-60 FPS on GPU, ~5-15 FPS on CPU
- **MobileNetV3**: ~100+ FPS on GPU, ~20-50 FPS on CPU
- **Total System**: ~5-30 FPS depending on hardware

## 🎯 Target Objects Detected

| Object | YOLO Detection | MobileNet Classification |
|--------|---------------|-------------------------|
| 📱 Phone | "cell phone" | "cellular_telephone" |
| 🖱️ Mouse | "cell phone" | "computer_mouse" |
| 🧢 Hat | "person" | "hat", "cap", "baseball_cap" |
| 🪥 Toothbrush | Not in YOLO | Requires custom training |

## 🔍 Debugging

### Check if Nodes are Running
```bash
ros2 node list
```

### Check Node Information
```bash
ros2 node info /yolo_detector
ros2 node info /mobilenet_classifier
ros2 node info /object_recognition_pipeline
```

### Monitor All Topics
```bash
ros2 topic list
```

## 🚁 Drone Integration

### Get Target Position for Navigation
```bash
ros2 topic echo /target_position
```

The `/target_position` topic provides:
- `x, y`: Image coordinates of target center
- `theta`: Detection confidence (0.0 to 1.0)

### Integration Example (Python)
```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D

class DroneController(Node):
    def __init__(self):
        super().__init__('drone_controller')
        self.target_sub = self.create_subscription(
            Pose2D,
            '/target_position',
            self.target_callback,
            10
        )
    
    def target_callback(self, msg):
        if msg.theta > 0.7:  # High confidence
            self.get_logger().info(f"Target at ({msg.x}, {msg.y}) with confidence {msg.theta}")
            # Send navigation commands to your drone
```

## 🛠️ Troubleshooting

### Common Issues

1. **"No module named 'ultralytics'"**
   ```bash
   source venv/bin/activate
   pip install ultralytics
   ```

2. **Camera not found**
   ```bash
   # Check available cameras
   ls /dev/video*
   
   # Test with different device
   ros2 launch drone_object_recognition test_with_usb_camera.launch.py camera_device:=/dev/video1
   ```

3. **Low FPS**
   ```bash
   # Lower confidence thresholds
   ros2 launch drone_object_recognition object_recognition.launch.py confidence_threshold:=0.3
   ```

4. **Nodes not starting**
   ```bash
   # Check if virtual environment is activated
   which python3
   
   # Should show: /home/aro/Documents/ObjectRec/venv/bin/python3
   ```

### Performance Tips

1. **Use GPU**: Ensure CUDA is installed for faster inference
2. **Lower Resolution**: Reduce camera resolution for higher FPS
3. **Adjust Thresholds**: Lower confidence thresholds for more detections
4. **Single Core**: Run on single CPU core for consistent timing

## 📱 Test with Real Objects

1. **Phone Test**: Point camera at a phone
2. **Mouse Test**: Point camera at a computer mouse
3. **Hat Test**: Point camera at someone wearing a hat
4. **Monitor**: Watch `/target_object_found` topic

## 🔄 System Architecture

```
Camera → YOLO Detector → MobileNet Classifier → Pipeline Controller
   ↓           ↓                ↓                    ↓
Image    Detections      Classifications       Status/Position
```

## 📞 Support

If you encounter issues:
1. Check this troubleshooting section
2. Review ROS2 logs: `ros2 node info <node_name>`
3. Test components individually: `python3 test_simple.py`
4. Verify virtual environment is activated

## 🎉 Success!

Your drone object recognition system is ready! The system can detect and classify objects in real-time, providing target positions for drone navigation.

Happy flying! 🚁✨
