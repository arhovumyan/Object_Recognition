# Video Recording and Analysis Guide

## Overview
The enhanced drone object recognition system now includes comprehensive video recording with detection overlays and detailed logging capabilities.

## Features Added

### 1. Video Recording with Detection Overlays
- **Real-time overlays**: Bounding boxes, labels, and confidence scores
- **Frame information**: Timestamp, frame number, FPS, detection count
- **Color-coded objects**: Different colors for different object types
- **Automatic saving**: Videos saved to `recordings/` directory with timestamps

### 2. Comprehensive Logging System
- **Detection logs**: Every detection with timestamp, confidence, bounding box
- **Session summaries**: Complete statistics and performance metrics
- **Multiple formats**: JSON and CSV for easy analysis
- **Real-time statistics**: FPS, detection counts, object type breakdowns

### 3. Analysis Tools
- **Results analyzer**: Python script to analyze detection logs
- **Visualization**: Timeline plots, distribution charts
- **Performance metrics**: Detection rates, confidence statistics

## Usage

### Recording Video with Detections

#### Start Recording System
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch drone_object_recognition realtime_phone_detection.launch.py record_video:=true
```

#### Parameters
- `record_video:=true` - Enable video recording (default: false)
- `camera_device:=/dev/video0` - Camera device path
- `image_width:=640` - Video width
- `image_height:=480` - Video height

### Output Files

#### Video Files
- **Location**: `recordings/`
- **Format**: MP4 with detection overlays
- **Naming**: `detection_recording_YYYYMMDD_HHMMSS.mp4`

#### Detection Logs
- **JSON format**: `detection_log_YYYYMMDD_HHMMSS.json`
- **CSV format**: `detection_log_YYYYMMDD_HHMMSS.csv`
- **Content**: Timestamp, frame number, class ID, confidence, bounding box, center coordinates

#### Session Summary
- **File**: `session_summary_YYYYMMDD_HHMMSS.json`
- **Content**: Session duration, total frames, FPS, detection statistics, camera settings

### Example Output Structure
```
recordings/
├── detection_recording_20251006_230830.mp4     # Video with overlays
├── detection_log_20251006_230839.json          # Detailed detection log
├── detection_log_20251006_230839.csv           # CSV format for analysis
└── session_summary_20251006_230839.json        # Session statistics
```

### Analyzing Results

#### Using the Analysis Script
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run drone_object_recognition analyze_results.py \
  --detection-log recordings/detection_log_20251006_230839.json \
  --session-summary recordings/session_summary_20251006_230839.json \
  --output-dir analysis_results
```

#### Analysis Output
- **Timeline plots**: Detection confidence over time
- **Distribution charts**: Object type breakdown
- **Performance metrics**: Detection rates, FPS statistics
- **Summary report**: Comprehensive analysis in text format

### Detection Statistics

#### Real-time Statistics
The system logs statistics every 100 frames:
```
Frame 100, FPS: 19.0, Total detections: 69, Stats: 0: 56, 67: 12, 65: 1
```

#### Object Type IDs
- `0`: person
- `67`: cell phone
- `65`: clock
- `79`: toothbrush
- `74`: mouse
- `44`: hat

### Video Overlay Information

#### Detection Overlays
- **Bounding boxes**: Colored rectangles around detected objects
- **Labels**: Object type and confidence score
- **Colors**: 
  - Green: person
  - Blue: cell phone
  - Yellow: toothbrush
  - Magenta: mouse
  - Orange: hat

#### Frame Information Overlay
- **Timestamp**: Current date and time
- **Frame number**: Sequential frame counter
- **FPS**: Current frames per second
- **Detection count**: Number of objects in current frame

### Performance Metrics

#### Typical Performance
- **FPS**: 15-20 FPS (depending on system)
- **Detection rate**: Varies based on scene content
- **Latency**: ~70ms inference time per frame

#### Session Statistics Example
```json
{
  "session_start": "2025-10-06T23:08:30.485103",
  "session_end": "2025-10-06T23:08:39.745228",
  "duration_seconds": 8.91,
  "total_frames": 173,
  "average_fps": 19.4,
  "total_detections": 141,
  "detection_statistics": {
    "0": 105,    // person
    "67": 34,    // cell phone
    "65": 2      // clock
  }
}
```

### Troubleshooting

#### Common Issues
1. **No detections logged**: Check topic names and ensure YOLO detector is running
2. **Video not recording**: Verify `record_video:=true` parameter
3. **Low FPS**: Reduce image resolution or check system resources
4. **Missing files**: Ensure `recordings/` directory exists and has write permissions

#### Log Analysis
- Check detection log files for detailed detection data
- Use session summary for overall performance metrics
- Analyze CSV files in spreadsheet applications for detailed analysis

### Integration with Drone Flight Controller

#### Topic Integration
The system publishes detection results on standard ROS2 topics:
- `/object_detections`: Detection2DArray messages
- `/target_position`: Pose2D messages for target objects
- `/camera/image_raw`: Raw camera feed

#### Flight Controller Integration
```python
# Example integration code
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from vision_msgs.msg import Detection2DArray

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
        # Implement drone navigation logic
        self.get_logger().info(f'Target at: x={msg.x}, y={msg.y}, theta={msg.theta}')
```

### Best Practices

#### Recording Sessions
1. **Test camera first**: Ensure camera is working before starting recording
2. **Monitor disk space**: Video files can be large
3. **Use appropriate resolution**: Balance quality vs performance
4. **Stop gracefully**: Use Ctrl+C to ensure proper file saving

#### Analysis Workflow
1. **Record session**: Capture video with detections
2. **Review logs**: Check detection statistics
3. **Analyze results**: Use analysis script for detailed insights
4. **Optimize parameters**: Adjust confidence thresholds based on results

### File Management

#### Cleanup
```bash
# Remove old recordings (optional)
find recordings/ -name "*.mp4" -mtime +7 -delete
find recordings/ -name "*.json" -mtime +7 -delete
find recordings/ -name "*.csv" -mtime +7 -delete
```

#### Backup
```bash
# Backup important recordings
tar -czf recordings_backup_$(date +%Y%m%d).tar.gz recordings/
```

This enhanced system provides comprehensive video recording and analysis capabilities for your drone object recognition system, making it easy to review, analyze, and optimize detection performance.
