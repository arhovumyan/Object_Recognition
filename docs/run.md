run commands for basic od without ros2:

source .venv/bin/activate
python3 live_object_detection.py

run commands with ros2:

cd /home/aro/Documents/ObjectRec/ros2_ws
export COLCON_TRACE="" && export AMENT_TRACE_SETUP_FILES="" && source install/setup.bash
ros2 launch drone_object_recognition object_recognition.launch.py