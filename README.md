# Drone Object Recognition System

A ROS2 Jazzy package for drone object recognition using YOLOv8s for object detection and MobileNetV3 for classification. The system is designed to detect and classify target objects: toothbrush, mouse (computer), phone, or hat.

## Features

- **YOLOv8s Object Detection**: Fast and accurate object detection
- **MobileNetV3 Classification**: Efficient classification of detected objects
- **ROS2 Integration**: Full ROS2 Jazzy compatibility with standard message types
- **Real-time Processing**: Optimized for drone applications
- **Configurable**: Easy to adjust thresholds and target objects
- **Debug Visualization**: Real-time debug images with bounding boxes

## System Architecture

```
Camera Input → YOLO Detector → MobileNet Classifier → Pipeline Controller
     ↓              ↓                ↓                    ↓
Image Topic → Detection Topic → Classification Topic → Status/Position Topics
```

## Prerequisites

### System Requirements
- Ubuntu 22.04 (recommended)
- ROS2 Jazzy
- Python 3.10+
- CUDA-capable GPU (optional, for faster inference)

### ROS2 Dependencies
```bash
sudo apt update
sudo apt install ros-jazzy-cv-bridge ros-jazzy-image-transport ros-jazzy-vision-msgs
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

## Installation

1. **Clone or copy the package to your ROS2 workspace:**
```bash
cd ~/ros2_ws/src
cp -r /path/to/ObjectRec/drone_object_recognition .
```

2. **Build the package:**
```bash
cd ~/ros2_ws
colcon build --packages-select drone_object_recognition
source install/setup.bash
```

3. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Launch the complete object recognition system:
```bash
ros2 launch drone_object_recognition object_recognition.launch.py
```

### Testing with USB Camera

For testing with a USB camera:
```bash
# Install usb_cam package first
sudo apt install ros-jazzy-usb-cam

# Launch with USB camera
ros2 launch drone_object_recognition test_with_usb_camera.launch.py
```

### YOLO Only (Detection without Classification)

For testing just the detection system:
```bash
ros2 launch drone_object_recognition yolo_only.launch.py
```

### Custom Parameters

Launch with custom parameters:
```bash
ros2 launch drone_object_recognition object_recognition.launch.py \
    camera_topic:=/my_camera/image_raw \
    confidence_threshold:=0.6 \
    classification_threshold:=0.4
```

## Topics

### Input Topics
- `/camera/image_raw` (sensor_msgs/Image): Camera input

### Output Topics
- `/object_detections` (vision_msgs/Detection2DArray): YOLO detection results
- `/object_classification` (std_msgs/String): MobileNet classification results
- `/target_object_found` (std_msgs/Bool): Whether target object is found
- `/recognition_status` (std_msgs/String): System status information
- `/target_position` (geometry_msgs/Pose2D): Position of found target object

### Debug Topics
- `/yolo_debug_image` (sensor_msgs/Image): YOLO detection visualization
- `/classification_debug_image` (sensor_msgs/Image): Classification visualization

### Command Topics
- `/recognition_command` (std_msgs/String): Commands to change target or control system

## Commands

Send commands to the recognition system:
```bash
# Change target object
ros2 topic pub /recognition_command std_msgs/String "data: 'phone'"

# Available targets: toothbrush, mouse, phone, hat
ros2 topic pub /recognition_command std_msgs/String "data: 'mouse'"

# Get status
ros2 topic pub /recognition_command std_msgs/String "data: 'status'"
```

## Configuration

### YOLO Configuration (`config/yolo_config.yaml`)
- Adjust confidence and IoU thresholds
- Modify target classes
- Enable/disable debug visualization

### MobileNet Configuration (`config/mobilenet_config.yaml`)
- Set classification thresholds
- Modify target object mappings
- Adjust model parameters

### Pipeline Configuration (`config/pipeline_config.yaml`)
- Set default target object
- Configure status update rates
- Modify topic names

## Target Objects

The system is designed to detect and classify these objects:

1. **Toothbrush**: Not directly in YOLO COCO dataset, requires custom training
2. **Mouse (Computer)**: Detected as "cell phone" by YOLO, classified by MobileNet
3. **Phone**: Detected as "cell phone" by YOLO, classified by MobileNet
4. **Hat**: Detected as "person" by YOLO, classified as hat-wearing person by MobileNet

## Performance Optimization

### GPU Acceleration
For faster inference, ensure CUDA is properly installed:
```bash
# Check CUDA installation
nvidia-smi

# PyTorch will automatically use GPU if available
```

### Model Optimization
- YOLOv8s is already optimized for speed vs accuracy
- MobileNetV3Small is efficient for classification
- Consider using TensorRT for further optimization on Jetson devices

## Troubleshooting

### Common Issues

1. **Model download fails**: Ensure internet connection for initial model download
2. **Low FPS**: Check GPU availability and reduce image resolution
3. **No detections**: Lower confidence thresholds in config files
4. **Camera not found**: Check camera permissions and device path

### Debug Commands
```bash
# Check if nodes are running
ros2 node list

# Monitor detection results
ros2 topic echo /object_detections

# Monitor classification results
ros2 topic echo /object_classification

# Check system status
ros2 topic echo /recognition_status
```

## Integration with Drone Control

The system publishes target positions that can be used for drone navigation:
```bash
# Monitor target positions
ros2 topic echo /target_position
```

The `target_position` topic contains:
- `x, y`: Image coordinates of target center
- `theta`: Detection confidence (0.0 to 1.0)

## Customization

### Adding New Target Objects
1. Update target mappings in `config/mobilenet_config.yaml`
2. Modify target lists in `scripts/object_recognition_pipeline.py`
3. Retrain YOLO model if needed for new object types

### Custom Model Training
For better accuracy on specific objects:
1. Collect training data for your specific objects
2. Fine-tune YOLOv8s on your dataset
3. Update model path in configuration files

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review ROS2 logs for error messages
3. Open an issue with detailed information about your setup
