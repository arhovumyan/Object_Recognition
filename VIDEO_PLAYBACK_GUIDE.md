# Video Playback Guide

## Problem Solved! ✅

The issue you encountered with VLC not being able to open the video files has been **fixed**. The problem was:

1. **Incompatible video codec**: The original system used a codec that created files VLC couldn't properly read
2. **Improper file finalization**: Video files weren't being properly closed and finalized

## What Was Fixed

### 1. Enhanced Video Codec Support
- **Multiple codec fallback**: The system now tries different codecs (XVID, MJPG, mp4v, H264, avc1) to ensure compatibility
- **Better codec selection**: Automatically selects the best available codec for your system
- **VLC compatibility**: Videos are now saved in formats that VLC can properly read

### 2. Improved File Handling
- **Proper file finalization**: Added proper cleanup and file finalization
- **No more temporary files**: Eliminated the `~` suffix issue that was causing VLC errors
- **Better error handling**: More robust video recording with proper error handling

## How to Play Your Recorded Videos

### Method 1: Using the Play Script (Recommended)

#### List Available Recordings
```bash
python3 ~/ros2_ws/src/drone_object_recognition/scripts/play_recording.py --list
```

#### Play Latest Recording
```bash
python3 ~/ros2_ws/src/drone_object_recognition/scripts/play_recording.py
```

#### Play Specific Recording
```bash
python3 ~/ros2_ws/src/drone_object_recognition/scripts/play_recording.py detection_recording_20251006_231558.mp4
```

### Method 2: Direct VLC Command

#### Find Your Recordings
```bash
ls -la ~/ros2_ws/recordings/
```

#### Play with VLC
```bash
vlc ~/ros2_ws/recordings/detection_recording_20251006_231558.mp4
```

### Method 3: Using ROS2 Command
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run drone_object_recognition play_recording.py --list
```

## What You'll See in the Videos

### Video Overlays
- **Bounding boxes**: Colored rectangles around detected objects
- **Object labels**: Object type and confidence score (e.g., "person: 0.94", "laptop: 0.87")
- **Frame information**: Timestamp, frame number, FPS, detection count
- **Color coding**:
  - 🟢 Green: person
  - 🔵 Blue: cell phone
  - 🟡 Yellow: toothbrush
  - 🟣 Magenta: mouse
  - 🟠 Orange: hat
  - ⚪ White: other objects

### Example Video Content
Your videos will show:
- Real-time object detection with bounding boxes
- Confidence scores for each detection
- Frame-by-frame statistics
- Timestamps for each frame
- Detection counts per frame

## File Information

### Video File Format
- **Format**: MP4 (ISO Media, MP4 Base Media v1)
- **Codec**: XVID/mp4v (VLC compatible)
- **Resolution**: 640x480 (configurable)
- **FPS**: ~19-20 FPS (varies by system performance)

### File Naming Convention
```
detection_recording_YYYYMMDD_HHMMSS.mp4
```
Example: `detection_recording_20251006_231558.mp4`

### File Sizes
- **Typical size**: 0.5-2 MB per 10 seconds of recording
- **Size depends on**: Resolution, frame rate, and detection activity

## Troubleshooting

### If VLC Still Can't Open Videos

#### Check File Format
```bash
file ~/ros2_ws/recordings/detection_recording_20251006_231558.mp4
```
Should show: `ISO Media, MP4 Base Media v1 [ISO 14496-12:2003]`

#### Try Different Players
```bash
# Try with mpv
mpv ~/ros2_ws/recordings/detection_recording_20251006_231558.mp4

# Try with ffplay
ffplay ~/ros2_ws/recordings/detection_recording_20251006_231558.mp4

# Try with totem (GNOME video player)
totem ~/ros2_ws/recordings/detection_recording_20251006_231558.mp4
```

#### Install VLC if Missing
```bash
sudo apt update
sudo apt install vlc
```

### If No Videos Are Created
1. **Check recording is enabled**: Make sure to use `record_video:=true`
2. **Check permissions**: Ensure `recordings/` directory is writable
3. **Check camera**: Verify camera is working before recording

## Recording New Videos

### Start Recording Session
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch drone_object_recognition realtime_phone_detection.launch.py record_video:=true
```

### Stop Recording
- Press `Ctrl+C` to stop the system gracefully
- Wait for "Video recording saved and finalized" message

### Verify Recording
```bash
# Check if file was created
ls -la ~/ros2_ws/recordings/

# Check file format
file ~/ros2_ws/recordings/detection_recording_*.mp4

# Play the video
python3 ~/ros2_ws/src/drone_object_recognition/scripts/play_recording.py
```

## Advanced Usage

### Custom Video Parameters
You can modify the video recording parameters in the launch file:
- `image_width`: Video width (default: 640)
- `image_height`: Video height (default: 480)
- `fps`: Frame rate (default: 30)

### Batch Playback
```bash
# Play all recordings in sequence
for file in ~/ros2_ws/recordings/detection_recording_*.mp4; do
    echo "Playing: $file"
    vlc "$file"
done
```

### Convert Videos (if needed)
```bash
# Convert to different format using ffmpeg
ffmpeg -i ~/ros2_ws/recordings/detection_recording_20251006_231558.mp4 \
       -c:v libx264 -c:a aac \
       ~/ros2_ws/recordings/converted_video.mp4
```

## Success Confirmation

Your videos should now:
- ✅ Open properly in VLC
- ✅ Show detection overlays
- ✅ Display frame information
- ✅ Play smoothly without errors
- ✅ Have proper MP4 format

The enhanced system ensures all recorded videos are compatible with standard media players and provide comprehensive logging of your drone's object detection performance!
