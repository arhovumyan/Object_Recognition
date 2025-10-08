#!/bin/bash
# Launch script for live object detection
# Automatically activates virtual environment and runs detection

cd "$(dirname "$0")"

echo "=============================================="
echo "  Live Object Detection Launcher"
echo "=============================================="
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

echo "Starting live object detection..."
echo ""
echo "NOTE: Currently running in CPU mode"
echo "      (GPU support for RTX 5070 not yet available in PyTorch)"
echo ""
echo "Press 'q' in the detection window to quit"
echo ""
echo "----------------------------------------------"
echo ""

python live_object_detection.py

echo ""
echo "Detection stopped."
