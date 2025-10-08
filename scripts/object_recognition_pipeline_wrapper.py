#!/usr/bin/env python3

import sys
import os

# Set up environment for ROS2 and virtual environment
os.environ['PYTHONPATH'] = '/opt/ros/jazzy/lib/python3.12/site-packages:' + os.environ.get('PYTHONPATH', '')
os.environ['LD_LIBRARY_PATH'] = '/opt/ros/jazzy/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['ROS_LOG_DIR'] = '/tmp/ros_logs'
os.environ['HOME'] = os.environ.get('HOME', '/tmp')

# Add the virtual environment path to sys.path
venv_python = "/home/aro/Documents/ObjectRec/.venv/bin/python3"
script_path = "/home/aro/Documents/ObjectRec/scripts/object_recognition_pipeline.py"

# Execute the actual script with the virtual environment Python
os.execv(venv_python, [venv_python, script_path] + sys.argv[1:])
