#!/usr/bin/env python3

"""
Quick System Test
Tests both standalone and ROS2 modes to identify issues
"""

import os
import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch imported successfully (version: {torch.__version__})")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO imported successfully")
    except ImportError as e:
        print(f"❌ Ultralytics import failed: {e}")
        return False
    
    try:
        import rclpy
        print("✅ ROS2 (rclpy) imported successfully")
    except ImportError as e:
        print(f"❌ ROS2 import failed: {e}")
        return False
    
    return True

def test_camera():
    """Test camera access"""
    print("\n📷 Testing camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera access successful")
                cap.release()
                return True
            else:
                print("❌ Camera opened but couldn't read frame")
                cap.release()
                return False
        else:
            print("❌ Could not open camera")
            return False
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_yolo_cpu():
    """Test YOLO with CPU only"""
    print("\n🤖 Testing YOLO (CPU mode)...")
    
    try:
        from ultralytics import YOLO
        import torch
        
        # Force CPU usage
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # Load model
        model = YOLO('models/yolov8n.pt')
        
        # Test with dummy image
        import numpy as np
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Run inference
        results = model(dummy_image, device='cpu', verbose=False)
        
        print("✅ YOLO CPU inference successful")
        return True
        
    except Exception as e:
        print(f"❌ YOLO test failed: {e}")
        return False

def test_ros2_basic():
    """Test basic ROS2 functionality"""
    print("\n🚀 Testing ROS2 basic functionality...")
    
    try:
        import rclpy
        from rclpy.node import Node
        
        # Initialize ROS2
        rclpy.init()
        
        # Create a simple node
        node = Node('test_node')
        node.destroy_node()
        
        # Shutdown ROS2
        rclpy.shutdown()
        
        print("✅ ROS2 basic functionality works")
        return True
        
    except Exception as e:
        print(f"❌ ROS2 test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 SYSTEM COMPREHENSIVE TEST")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run tests
    if not test_imports():
        all_tests_passed = False
    
    if not test_camera():
        all_tests_passed = False
    
    if not test_yolo_cpu():
        all_tests_passed = False
    
    if not test_ros2_basic():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✅ ALL TESTS PASSED!")
        print("Your system is ready for object detection.")
        print("\n🚀 Try running:")
        print("   python3 src/live_object_detection.py")
    else:
        print("❌ SOME TESTS FAILED")
        print("Check the errors above and fix them before proceeding.")
        print("\n🔧 Quick fixes:")
        print("   - Install missing packages: pip install -r requirements.txt")
        print("   - Check camera permissions")
        print("   - For CUDA issues, try CPU mode first")

if __name__ == "__main__":
    main()
