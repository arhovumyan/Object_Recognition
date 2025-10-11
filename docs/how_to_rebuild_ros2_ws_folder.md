How to Rebuild When Needed
When you're ready for production or want to test with ROS2:
1. Create the ROS2 workspace structure:

cd /home/aro/Documents/ObjectRec
mkdir -p ros2_ws/src

2. Copy your package to the workspace:

cp -r src ros2_ws/src/drone_object_recognition
cp -r launch ros2_ws/src/drone_object_recognition/
cp -r models ros2_ws/src/drone_object_recognition/
cp package.xml ros2_ws/src/drone_object_recognition/
cp requirements.txt ros2_ws/src/drone_object_recognition/

3. Build the ROS2 package:

cd ros2_ws
colcon build
source install/setup.bash

4. Run it:

ros2 launch drone_object_recognition object_recognition.launch.py