#!/usr/bin/env python3

"""
CUDA Compatibility Fix Script
Diagnoses and fixes CUDA compatibility issues with PyTorch and YOLOv8
"""

import torch
import sys
import subprocess
import os

def check_system_info():
    """Check system information"""
    print("🔍 SYSTEM DIAGNOSTICS")
    print("=" * 50)
    
    # Python and PyTorch info
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version (PyTorch): {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Compute capability: {props.major}.{props.minor}")
    else:
        print("❌ CUDA not available")
    
    print()

def test_cuda_compatibility():
    """Test CUDA compatibility"""
    print("🧪 CUDA COMPATIBILITY TEST")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - running CPU-only mode")
        return False
    
    try:
        # Test basic CUDA operations
        print("Testing basic CUDA operations...")
        x = torch.randn(10, 10).cuda()
        y = torch.randn(10, 10).cuda()
        z = torch.mm(x, y)
        print("✅ Basic CUDA operations work")
        
        # Test model loading on GPU
        print("Testing model loading on GPU...")
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        model.to('cuda')
        print("✅ YOLO model loads on GPU")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA compatibility test failed: {e}")
        return False

def fix_cuda_issues():
    """Provide solutions for CUDA issues"""
    print("🔧 CUDA COMPATIBILITY FIXES")
    print("=" * 50)
    
    print("Common solutions for 'no kernel image available' error:")
    print()
    print("1️⃣  REINSTALL PYTORCH WITH CORRECT CUDA VERSION:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print()
    print("2️⃣  FORCE CPU MODE (Quick Fix):")
    print("   export CUDA_VISIBLE_DEVICES=''")
    print("   python3 src/live_object_detection.py")
    print()
    print("3️⃣  USE CPU-ONLY YOLO MODEL:")
    print("   # Modify your script to use CPU by default")
    print()
    print("4️⃣  CHECK GPU COMPUTE CAPABILITY:")
    print("   # Your GPU might be too old for current PyTorch CUDA kernels")
    print()

def create_cpu_only_script():
    """Create a CPU-only version of the detection script"""
    print("📝 CREATING CPU-ONLY SCRIPT")
    print("=" * 50)
    
    cpu_script = """#!/usr/bin/env python3

'''
CPU-Only Object Detection Script
Forces CPU usage to avoid CUDA compatibility issues
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import cv2
import numpy as np
from ultralytics import YOLO
import time

def main():
    print("🚀 CPU-Only Object Detection")
    print("=" * 40)
    
    # Force CPU usage
    import torch
    print(f"PyTorch device: CPU (forced)")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Load YOLO model (will use CPU automatically)
    print("Loading YOLOv8n model...")
    model = YOLO('models/yolov8n.pt')
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Camera opened successfully")
    print("Press 'q' to quit")
    
    fps_counter = 0
    fps_start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection (CPU-only)
        results = model(frame, conf=0.3, verbose=False, device='cpu')
        
        # Draw results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{class_name} {confidence:.1f}"
                    cv2.putText(frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate and display FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
            
            fps_text = f"FPS: {fps} (CPU)"
            cv2.putText(frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display frame
        cv2.imshow('CPU-Only Object Detection', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done!")

if __name__ == "__main__":
    main()
"""
    
    with open('/home/aro/Documents/ObjectRec/src/live_object_detection_cpu.py', 'w') as f:
        f.write(cpu_script)
    
    print("✅ Created CPU-only script: src/live_object_detection_cpu.py")
    print("   Run with: python3 src/live_object_detection_cpu.py")

def main():
    """Main function"""
    print("🚀 CUDA COMPATIBILITY DIAGNOSTIC TOOL")
    print("=" * 60)
    print()
    
    check_system_info()
    
    if test_cuda_compatibility():
        print("✅ CUDA compatibility test passed!")
        print("Your system should work with GPU acceleration.")
    else:
        print("❌ CUDA compatibility issues detected")
        fix_cuda_issues()
        create_cpu_only_script()
        
        print("\n🎯 RECOMMENDED NEXT STEPS:")
        print("1. Try the CPU-only script: python3 src/live_object_detection_cpu.py")
        print("2. If you need GPU, reinstall PyTorch with correct CUDA version")
        print("3. For ROS2, use CPU mode or fix CUDA compatibility first")

if __name__ == "__main__":
    main()
