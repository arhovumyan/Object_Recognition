# 🧪 ROS2 YOLO Detection System Testing Guide

This guide shows you how to test the ROS2 YOLO detection system with different approaches and scenarios.

## 🚀 Quick Start Testing

### **Method 1: Interactive Test Suite**
```bash
cd /home/aro/Documents/ObjectRec
python3 scripts/test_ros2_yolo.py
```
This provides an interactive menu to test different components.

### **Method 2: Individual Component Tests**
```bash
# Test Mock MAVROS Services
cd /home/aro/Documents/ObjectRec/ros2_ws
bash -c "source install/setup.bash && ros2 run drone_object_recognition mock_mavros_services.py"

# Test YOLO Data Publisher  
bash -c "source install/setup.bash && ros2 run drone_object_recognition test_yolo_publisher.py"

# Test YOLO Detection Node
bash -c "source install/setup.bash && ros2 run drone_object_recognition yolo_detection_ros2.py"
```

### **Method 3: Full System Integration**
```bash
cd /home/aro/Documents/ObjectRec/ros2_ws
bash -c "source install/setup.bash && ros2 launch drone_object_recognition test_yolo_detection.launch.py"
```

## 🔍 Testing Scenarios

### **Scenario 1: Mock Drone Environment**
Test the system without a real drone using mock services:

```bash
# Terminal 1: Start Mock MAVROS Services
cd /home/aro/Documents/ObjectRec/ros2_ws
bash -c "source install/setup.bash && ros2 run drone_object_recognition mock_mavros_services.py"

# Terminal 2: Start YOLO Detection Node
cd /home/aro/Documents/ObjectRec/ros2_ws
bash -c "source install/setup.bash && ros2 run drone_object_recognition yolo_detection_ros2.py"

# Terminal 3: Start Test YOLO Publisher
cd /home/aro/Documents/ObjectRec/ros2_ws
bash -c "source install/setup.bash && ros2 run drone_object_recognition test_yolo_publisher.py"
```

### **Scenario 2: With Real Camera**
Test with your webcam for real object detection:

```bash
cd /home/aro/Documents/ObjectRec/ros2_ws
bash -c "source install/setup.bash && ros2 launch drone_object_recognition object_recognition.launch.py"
```

### **Scenario 3: YOLO Detection Only**
Test just the YOLO detection without camera:

```bash
cd /home/aro/Documents/ObjectRec/ros2_ws
bash -c "source install/setup.bash && ros2 launch drone_object_recognition test_yolo_detection.launch.py use_camera:=false"
```

## 📊 Monitoring and Debugging

### **Check Available Topics**
```bash
cd /home/aro/Documents/ObjectRec/ros2_ws
bash -c "source install/setup.bash && ros2 topic list"
```

### **Monitor YOLO Detection Results**
```bash
cd /home/aro/Documents/ObjectRec/ros2_ws
bash -c "source install/setup.bash && ros2 topic echo /yolo_result"
```

### **Monitor MAVROS Messages**
```bash
cd /home/aro/Documents/ObjectRec/ros2_ws
bash -c "source install/setup.bash && ros2 topic echo /mavros/global_position/global"
```

### **Check Service Availability**
```bash
cd /home/aro/Documents/ObjectRec/ros2_ws
bash -c "source install/setup.bash && ros2 service list"
```

## 🎯 Expected Behavior

### **When Everything Works:**
1. **Mock MAVROS Services**: Publishes GPS data and provides service endpoints
2. **YOLO Publisher**: Publishes mock detection results every 1 second
3. **YOLO Detection Node**: 
   - Receives detection data
   - Calculates GPS coordinates
   - Adds objects to waypoint queue
   - Publishes status messages
4. **Camera Publisher**: Provides live camera feed (if enabled)

### **Sample Output:**
```
[INFO] [mock_mavros_services]: Mock MAVROS Services started
[INFO] [test_yolo_publisher]: Detected: PERSON (simulated)
[INFO] [yolo_result_subscriber]: 1 object(s) detected!
[INFO] [yolo_result_subscriber]: Calculated Position: LAT: 38.3149, LONG:-76.5501
[INFO] [yolo_result_subscriber]: Object 'person' added at LAT: 38.3149, LONG: -76.5501, ALT: 20
```

## 🔧 Troubleshooting

### **Common Issues:**

1. **"Service not available" errors**
   - Make sure Mock MAVROS Services is running first
   - Check with: `ros2 service list`

2. **"No objects detected"**
   - Verify YOLO Publisher is running
   - Check topic: `ros2 topic echo /yolo_result`

3. **Camera not working**
   - Test camera with: `python3 src/live_object_detection.py`
   - Check camera permissions

4. **Package not found**
   - Rebuild: `cd ros2_ws && colcon build`
   - Source: `source install/setup.bash`

### **Debug Commands:**
```bash
# Check node status
ros2 node list

# Check topic rates
ros2 topic hz /yolo_result

# Check node info
ros2 node info /yolo_result_subscriber
```

## 🎮 Advanced Testing

### **Custom Detection Scenarios:**
Modify `test_yolo_publisher.py` to test specific scenarios:
```python
# In the parameters, set:
'detect_person': True,
'detect_car': True,
'detect_bottle': False
```

### **Performance Testing:**
```bash
# Monitor system resources
htop

# Check ROS2 performance
ros2 topic hz /yolo_result
ros2 topic bw /yolo_result
```

### **Integration with Real MAVROS:**
Replace mock services with real MAVROS:
```bash
# Start real MAVROS (if available)
ros2 launch mavros px4.launch

# Then run YOLO detection
ros2 run drone_object_recognition yolo_detection_ros2.py
```

## 📝 Test Results Logging

The system automatically logs:
- Detection results to console
- Photos to `detected_objects/` directory
- Status messages to MAVROS
- Performance metrics

Check logs in:
- Console output
- ROS2 logs: `~/.ros/log/`
- Detection photos: `install/drone_object_recognition/share/drone_object_recognition/detected/`

## 🎯 Success Criteria

✅ **System is working correctly when:**
- Mock services respond to requests
- YOLO publisher generates detection data
- Detection node processes results and calculates GPS
- Objects are added to waypoint queue
- Status messages are published
- Photos are saved for detected objects

This testing framework allows you to verify the entire YOLO detection pipeline without needing a real drone!
