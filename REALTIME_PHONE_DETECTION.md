# 📱 Real-Time Phone Detection System

## 🎯 Overview

This enhanced system opens your USB camera, detects objects in real-time using YOLOv8s, saves video recordings, and uses MobileNetV3 to specifically classify whether detected objects are phones or not.

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd /home/aro/Documents/ObjectRec
source venv/bin/activate
```

### 2. Build ROS2 Package
```bash
cd ~/ros2_ws
colcon build --packages-select drone_object_recognition
source install/setup.bash
```

### 3. Run Real-Time Phone Detection
```bash
ros2 launch drone_object_recognition realtime_phone_detection.launch.py
```

### 4. Monitor Results (Optional)
```bash
# In another terminal
ros2 run drone_object_recognition phone_detection_monitor.py
```

## 📊 What the System Does

### 🔍 **YOLOv8s Detection**
- Opens USB camera (default: `/dev/video0`)
- Detects objects in real-time at 30 FPS
- **Focuses on phone detection** with lower confidence threshold (0.3)
- Highlights phones with **yellow bounding boxes**
- Logs detection statistics

### 🧠 **MobileNetV3 Classification**
- Takes detected objects and classifies them
- **Specifically identifies phones** with keywords:
  - `cellular_telephone`
  - `cellular_phone`
  - `cellphone`
  - `mobile_phone`
  - `telephone`
- Uses lower threshold (0.15) for better phone detection
- Highlights confirmed phones with **bright yellow boxes**

### 📹 **Video Recording**
- Automatically saves video to `recordings/` folder
- Filename format: `detection_recording_YYYYMMDD_HHMMSS.mp4`
- Records all detection results with bounding boxes

### 📈 **Real-Time Monitoring**
- Live video feed with detection overlays
- Statistics display (FPS, detection count, phone rate)
- Real-time phone detection notifications

## 🎮 Usage Examples

### Basic Phone Detection
```bash
ros2 launch drone_object_recognition realtime_phone_detection.launch.py
```

### Custom Camera Settings
```bash
ros2 launch drone_object_recognition realtime_phone_detection.launch.py \
    camera_device:=/dev/video1 \
    camera_width:=1280 \
    camera_height:=720 \
    fps:=15
```

### Adjust Detection Sensitivity
```bash
ros2 launch drone_object_recognition realtime_phone_detection.launch.py \
    confidence_threshold:=0.2 \
    classification_threshold:=0.1
```

### Disable Video Recording
```bash
ros2 launch drone_object_recognition realtime_phone_detection.launch.py \
    record_video:=false
```

## 📱 Phone Detection Features

### 🎯 **Optimized for Phones**
- **Lower detection threshold** for phones (0.3 vs 0.5)
- **Special phone classification** with lower threshold (0.15)
- **Bright yellow highlighting** for confirmed phones
- **Real-time phone notifications** in logs

### 📊 **Detection Statistics**
- Tracks total detections vs phone detections
- Calculates phone detection rate percentage
- Logs statistics every 50 detections
- Real-time FPS monitoring

### 🔍 **Visual Indicators**
- **Yellow boxes**: YOLO phone detections
- **Bright yellow boxes**: Confirmed MobileNet phone classifications
- **Green boxes**: Other target objects
- **Blue boxes**: Non-target objects

## 📹 Video Recording

### 📁 **Recording Location**
- Videos saved in `recordings/` folder
- Automatic folder creation
- Timestamped filenames

### 🎬 **Recording Content**
- Original camera feed
- All detection bounding boxes
- Classification results
- Real-time processing

### 💾 **File Format**
- **Format**: MP4 (mp4v codec)
- **Resolution**: Matches camera settings
- **FPS**: Matches camera FPS
- **Quality**: High quality with detection overlays

## 🖥️ Monitor Interface

### 📺 **Live Display**
- Real-time video feed
- Detection overlays
- Statistics overlay
- Phone detection indicators

### ⌨️ **Keyboard Controls**
- **'q'**: Quit monitor
- **'s'**: Show detailed statistics
- **ESC**: Close display window

### 📊 **Statistics Display**
- Current FPS
- Total frames processed
- Total detections
- Phone detections count
- Phone detection rate percentage

## 📡 ROS2 Topics

### 📥 **Input Topics**
- `/camera/image_raw`: Camera feed from USB camera

### 📤 **Output Topics**
- `/object_detections`: YOLO detection results
- `/object_classification`: MobileNet classification results
- `/target_object_found`: Phone detection status
- `/recognition_status`: System status
- `/yolo_debug_image`: YOLO detection visualization
- `/classification_debug_image`: Classification visualization

## 🔧 Configuration

### 📷 **Camera Settings**
```yaml
camera_device: '/dev/video0'    # USB camera path
camera_width: 640               # Image width
camera_height: 480              # Image height
fps: 30                         # Frame rate
record_video: true              # Enable recording
```

### 🎯 **Detection Settings**
```yaml
confidence_threshold: 0.3       # YOLO confidence (lower for phones)
classification_threshold: 0.15  # MobileNet threshold (lower for phones)
```

## 🚨 Troubleshooting

### 📷 **Camera Issues**
```bash
# Check available cameras
ls /dev/video*

# Test different camera device
ros2 launch drone_object_recognition realtime_phone_detection.launch.py camera_device:=/dev/video1
```

### 🐌 **Performance Issues**
```bash
# Lower FPS for better performance
ros2 launch drone_object_recognition realtime_phone_detection.launch.py fps:=15

# Lower resolution
ros2 launch drone_object_recognition realtime_phone_detection.launch.py \
    camera_width:=320 camera_height:=240
```

### 📱 **Phone Detection Issues**
```bash
# Lower thresholds for more sensitive detection
ros2 launch drone_object_recognition realtime_phone_detection.launch.py \
    confidence_threshold:=0.2 \
    classification_threshold:=0.1
```

### 🎥 **Recording Issues**
```bash
# Check recordings folder
ls -la recordings/

# Test without recording
ros2 launch drone_object_recognition realtime_phone_detection.launch.py record_video:=false
```

## 📊 Expected Performance

### ⚡ **Speed**
- **YOLO Detection**: ~30-60 FPS (GPU), ~5-15 FPS (CPU)
- **MobileNet Classification**: ~100+ FPS (GPU), ~20-50 FPS (CPU)
- **Total System**: ~5-30 FPS depending on hardware

### 💾 **Storage**
- **Video Size**: ~10-50 MB per minute (depends on resolution/FPS)
- **Recording Quality**: High quality with detection overlays

### 🎯 **Accuracy**
- **Phone Detection**: Optimized with lower thresholds
- **False Positives**: Minimal with dual-stage detection
- **Real-time Processing**: Continuous detection and classification

## 🔄 System Flow

```
USB Camera → YOLO Detection → MobileNet Classification → Video Recording
     ↓              ↓                    ↓                    ↓
Live Feed → Phone Candidates → Phone Confirmation → Saved Video
```

## 📝 Log Output Example

```
[INFO] Camera opened successfully: /dev/video0
[INFO] Resolution: 640x480, FPS: 30.0
[INFO] Video recording started: recordings/detection_recording_20241201_143022.mp4
[INFO] 📱 PHONE DETECTED: cellular_telephone (confidence: 0.847)
[INFO] Detection stats - Total: 150, Phones: 23 (15.3%)
[INFO] Classification stats - Total: 45, Phones: 12 (26.7%)
```

## 🎉 Success Indicators

✅ **Camera opens successfully**  
✅ **Video recording starts**  
✅ **Yellow boxes appear around phones**  
✅ **"📱 PHONE DETECTED" messages in logs**  
✅ **Statistics show phone detection rate**  
✅ **Video files saved in recordings folder**  

Your real-time phone detection system is now ready! Point your camera at phones and watch the magic happen! 📱✨
