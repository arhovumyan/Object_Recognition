# GitHub Repository Setup Guide

## 🚀 Quick Start for GitHub Users

### Prerequisites
- Ubuntu 20.04+ or compatible Linux distribution
- Python 3.8+
- ROS2 Jazzy installed
- Git installed

### 1. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/drone-object-recognition.git
cd drone-object-recognition

# Make setup script executable
chmod +x setup.sh

# Run the automated setup
./setup.sh
```

### 2. Activate Environment
```bash
# Activate the virtual environment
source venv/bin/activate
# OR use the activation script
./activate_env.sh
```

### 3. Test the System
```bash
# Test individual components
python3 test_simple.py

# Build ROS2 package
cd ~/ros2_ws
colcon build --packages-select drone_object_recognition
source install/setup.bash
```

### 4. Run the System
```bash
# Standard quality recording
ros2 launch drone_object_recognition realtime_phone_detection.launch.py record_video:=true

# High quality recording
ros2 launch drone_object_recognition high_quality_recording.launch.py record_video:=true
```

## 📁 What's Included

### Core Files
- `scripts/` - Python scripts for YOLO detection, MobileNet classification, and camera recording
- `launch/` - ROS2 launch files for different configurations
- `config/` - YAML configuration files
- `drone_object_recognition/` - ROS2 Python package

### Documentation
- `README.md` - Main project documentation
- `USAGE_GUIDE.md` - Comprehensive usage instructions
- `VIDEO_RECORDING_GUIDE.md` - Video recording and analysis guide
- `HIGH_QUALITY_GUIDE.md` - High-quality recording setup
- `QUICK_START.md` - Quick start instructions

### Setup Scripts
- `setup.sh` - Automated setup script
- `download_models.sh` - Model download script
- `activate_env.sh` - Environment activation helper

## 🔧 What's Auto-Downloaded

The following files are automatically downloaded during setup:
- **YOLOv8s model** (~22MB) - Downloaded on first run
- **MobileNetV3 weights** - Downloaded by TensorFlow
- **Python dependencies** - Installed via pip

## 📊 System Requirements

### Minimum Requirements
- **CPU**: 4 cores, 2.0 GHz
- **RAM**: 8 GB
- **Storage**: 5 GB free space
- **Camera**: USB webcam

### Recommended Requirements
- **CPU**: 8 cores, 3.0 GHz
- **RAM**: 16 GB
- **Storage**: 10 GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional)

## 🎯 Features

### Object Detection
- **YOLOv8s**: Fast, accurate object detection
- **Target Objects**: person, cell phone, toothbrush, mouse, hat
- **Real-time Processing**: 15-30 FPS depending on hardware

### Video Recording
- **High Quality**: HD (1280x720) and Full HD (1920x1080) support
- **Detection Overlays**: Real-time bounding boxes and labels
- **Comprehensive Logging**: JSON/CSV detection logs
- **VLC Compatible**: Plays in all major media players

### Analysis Tools
- **Results Analyzer**: Python script for detailed analysis
- **Visualization**: Timeline plots and distribution charts
- **Performance Metrics**: FPS, detection rates, confidence statistics

## 🔍 Troubleshooting

### Common Issues

#### Setup Fails
```bash
# Fix Python environment issues
./fix_environment.sh

# Manual virtual environment setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Camera Not Found
```bash
# Check available cameras
ls /dev/video*

# Install USB camera support
sudo apt install ros-jazzy-usb-cam
```

#### Models Not Downloading
```bash
# Manual model download
./download_models.sh

# Or run setup again
./setup.sh
```

### Performance Issues

#### Low FPS
- Reduce camera resolution: `camera_width:=640 camera_height:=480`
- Lower detection confidence: `confidence_threshold:=0.5`
- Close other applications

#### High CPU Usage
- Use GPU acceleration if available
- Reduce processing frequency
- Optimize camera settings

## 📈 Performance Tips

### For Best Performance
1. **Use SSD storage** for faster model loading
2. **Close unnecessary applications** to free CPU/RAM
3. **Use appropriate resolution** for your hardware
4. **Enable GPU acceleration** if available

### For Best Quality
1. **Use Full HD resolution** (1920x1080)
2. **Ensure good lighting** for better detection
3. **Use stable camera mount** to reduce motion blur
4. **Position objects clearly** in camera view

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style
- Follow Python PEP 8 guidelines
- Add docstrings to functions
- Include type hints where possible
- Update documentation for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Documentation
- Check the guides in the repository
- Read the README.md for basic setup
- Consult USAGE_GUIDE.md for detailed instructions

### Issues
- Check existing GitHub issues
- Create a new issue with detailed description
- Include system information and error logs

### Community
- Join discussions in GitHub Discussions
- Share your results and improvements
- Help other users with their setup

## 🎉 Success!

Once setup is complete, you should have:
- ✅ Working drone object recognition system
- ✅ High-quality video recording capabilities
- ✅ Comprehensive detection logging
- ✅ Analysis tools for reviewing results
- ✅ Integration-ready for drone flight controllers

Happy flying! 🚁📱
