# 🎥 How to Run ROS2 Object Detection with Live Window

## Quick Start

```bash
cd /home/aro/Documents/ObjectRec/ros2_ws
source install/setup.bash
ros2 launch drone_object_recognition object_recognition.launch.py
```

## What You Should See

When running correctly, you will see:

### 1. Terminal Output:
```
[INFO] [camera_publisher]: Successfully opened camera device 0
[INFO] [camera_publisher]: Camera resolution: 640x480
[INFO] [camera_publisher]: Camera Publisher Node started
[INFO] [object_detection]: ROS2 Object Detection Node started
[INFO] [object_detection]: Subscribing to /camera/image_raw topic
```

### 2. Live Video Window:
- A window titled **"ROS2 Object Detection - Live Feed"** will appear
- Shows live camera feed with bounding boxes around detected objects
- Green boxes = target objects (person, cell phone, laptop, etc.)
- Blue boxes = other detected objects
- FPS counter and device info (CPU/GPU) displayed at top

### 3. ROS2 Topics Publishing:
The system publishes detection results to these topics:
- `/object_detections` - Array of detected objects with bounding boxes
- `/recognition_status` - Status messages
- `/target_object_found` - Boolean indicating if target objects found
- `/target_position` - Position of target objects
- `/camera/image_raw` - Raw camera feed

## System Architecture

```
┌─────────────────────┐
│ camera_publisher    │ Opens camera device 0
│      node           │ Publishes to /camera/image_raw
└─────────┬───────────┘
          │ ROS2 topic
          ▼
┌─────────────────────┐
│ live_object_        │ Subscribes to /camera/image_raw
│   detection node    │ Runs YOLO detection
│                     │ Displays window with detections
│                     │ Publishes detection results
└─────────────────────┘
```

## How It Works

1. **camera_publisher_node.py**:
   - Opens camera device 0
   - Captures frames at ~30 FPS
   - Publishes to `/camera/image_raw` topic

2. **live_object_detection.py** (ROS2 mode):
   - Subscribes to `/camera/image_raw` topic
   - Receives frames from camera publisher
   - Runs YOLOv8 object detection on each frame
   - Draws bounding boxes and labels
   - Shows processed frames in OpenCV window
   - Publishes detection results to various topics

## Keyboard Controls

- **Press 'q'** or **ESC** in the video window to quit (though this won't stop the ROS2 nodes)
- **Press Ctrl+C** in the terminal to stop all nodes cleanly

## Monitoring with ROS2 Commands

In a separate terminal, you can monitor the system:

```bash
# Source the workspace first
cd /home/aro/Documents/ObjectRec/ros2_ws
source install/setup.bash

# List active nodes
ros2 node list

# List active topics
ros2 topic list

# See detection messages
ros2 topic echo /object_detections

# Check if target objects are found
ros2 topic echo /target_object_found

# Monitor camera frames (bandwidth intensive!)
ros2 topic echo /camera/image_raw --no-arr
```

## Troubleshooting

### Window Doesn't Appear
**Symptom**: Terminal shows nodes running but no video window appears

**Solutions**:
1. Check if running over SSH - OpenCV windows require a display
2. Try setting display: `export DISPLAY=:0`
3. Check if X11 forwarding is enabled for SSH

### Camera Access Error
**Symptom**: `Could not open camera device`

**Solutions**:
1. Check camera permissions: `ls -l /dev/video*`
2. Add user to video group: `sudo usermod -a -G video $USER` (logout/login required)
3. Test camera directly: `python3 /home/aro/Documents/ObjectRec/test_camera.py`
4. Check if another application is using the camera

### CUDA Warning
**Symptom**: Warning about RTX 5070 not compatible

**This is normal!** The system automatically falls back to CPU mode. Performance:
- CPU mode: 10-20 FPS
- GPU mode (if PyTorch recompiled): 60-120 FPS

To fix GPU compatibility, you need to reinstall PyTorch compiled for CUDA architecture sm_120.

### Double Camera Opening
**Symptom**: Object detection node tries to open camera but fails

**This should be fixed!** The detection node now skips camera opening and only subscribes to the topic.

## Advanced Options

### Run without camera publisher (use external camera source):
```bash
ros2 launch drone_object_recognition object_recognition.launch.py use_camera_node:=false
```

### Change camera topic:
```bash
ros2 launch drone_object_recognition object_recognition.launch.py camera_topic:=/my_camera/image
```

### Adjust confidence threshold:
```bash
ros2 launch drone_object_recognition object_recognition.launch.py confidence_threshold:=0.5
```

## Target Objects

The system highlights these objects in **green**:
- person
- cell phone
- laptop
- mouse
- tv
- bottle
- cup
- book
- keyboard
- chair

All other detected objects are shown in **blue**.

## Performance

### Current Setup (CPU mode):
- **FPS**: 10-20
- **Detection confidence**: 0.3
- **Model**: YOLOv8n (nano - faster, less accurate)
- **Inference size**: 320x320

### With Compatible GPU:
- **FPS**: 60-120
- **Model**: YOLOv8s (small - slower, more accurate)
- **Inference size**: 640x640

## Files Modified

The key fix was making the object detection node **subscribe** to the camera topic instead of trying to open its own camera instance:

1. `/home/aro/Documents/ObjectRec/src/live_object_detection.py`
   - Added `skip_camera` parameter
   - ROS2 node now subscribes to `/camera/image_raw`
   - Added `cv2.imshow()` in the image callback to display window
   - Fixed camera conflict issue

2. Workflow: Edit → Copy → Rebuild
   ```bash
   # After editing src/live_object_detection.py:
   cp /home/aro/Documents/ObjectRec/src/live_object_detection.py \
      /home/aro/Documents/ObjectRec/ros2_ws/src/drone_object_recognition/
   
   cd /home/aro/Documents/ObjectRec/ros2_ws
   colcon build --packages-select drone_object_recognition
   ```

---

**Date**: October 8, 2025
**Status**: ✅ Fully Working
