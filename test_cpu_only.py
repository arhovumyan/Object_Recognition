#!/usr/bin/env python3

"""
CPU-only test script for the object recognition system.
This script tests all components in CPU-only mode for systems without GPU support.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image as PILImage
import os
import torch


def test_yolo_cpu():
    """Test YOLOv8s model in CPU-only mode."""
    print("Testing YOLOv8s (CPU-only)...")
    
    try:
        # Load model and force CPU usage
        model = YOLO('yolov8s.pt')
        model.to('cpu')
        print("✅ YOLOv8s model loaded successfully (CPU mode)")
        
        # Create a test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run inference with CPU device
        results = model(test_image, conf=0.5, device='cpu')
        print("✅ YOLO inference completed (CPU)")
        
        # Print available classes
        print(f"Available YOLO classes: {len(model.names)} classes")
        target_classes = ['person', 'cell phone', 'laptop', 'tv']
        for cls in target_classes:
            if cls in model.names.values():
                class_id = list(model.names.keys())[list(model.names.values()).index(cls)]
                print(f"   - {cls}: class_id {class_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO test failed: {str(e)}")
        return False


def test_mobilenet_cpu():
    """Test MobileNetV3 model in CPU-only mode."""
    print("\nTesting MobileNetV3 (CPU-only)...")
    
    try:
        # Force TensorFlow to use CPU only
        tf.config.set_visible_devices([], 'GPU')
        print("✅ TensorFlow configured for CPU-only mode")
        
        # Load model
        model = MobileNetV3Small(
            input_shape=(224, 224, 3),
            weights='imagenet',
            include_top=True
        )
        print("✅ MobileNetV3 model loaded successfully (CPU mode)")
        
        # Create a test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Preprocess
        img_array = image.img_to_array(test_image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Run inference with CPU device
        with tf.device('/CPU:0'):
            predictions = model.predict(img_array, verbose=0)
        print("✅ MobileNetV3 inference completed (CPU)")
        
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        
        print("Sample predictions:")
        for class_id, class_name, confidence in decoded_predictions:
            print(f"   - {class_name}: {confidence:.3f}")
        
        # Check for target objects
        target_keywords = ['cellular_telephone', 'mouse', 'hat', 'cap']
        found_targets = []
        for class_id, class_name, confidence in decoded_predictions:
            for keyword in target_keywords:
                if keyword in class_name.lower():
                    found_targets.append(class_name)
        
        if found_targets:
            print(f"Found potential target objects: {found_targets}")
        else:
            print("No direct target objects in top predictions (this is normal for random images)")
        
        return True
        
    except Exception as e:
        print(f"❌ MobileNetV3 test failed: {str(e)}")
        return False


def test_opencv():
    """Test OpenCV functionality."""
    print("\nTesting OpenCV...")
    
    try:
        # Test basic OpenCV functionality
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), 2)
        
        # Test image operations
        resized = cv2.resize(test_image, (224, 224))
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        print("✅ OpenCV basic operations working")
        print(f"Image shape: {test_image.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenCV test failed: {str(e)}")
        return False


def test_dependencies():
    """Test all required dependencies."""
    print("Testing dependencies...")
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('tensorflow', 'TensorFlow'),
        ('cv2', 'OpenCV'),
        ('ultralytics', 'Ultralytics YOLO'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy')
    ]
    
    all_good = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - not installed")
            all_good = False
    
    return all_good


def main():
    """Run all tests in CPU-only mode."""
    print("Drone Object Recognition System - CPU-Only Component Tests")
    print("=" * 70)
    print("This test runs all components in CPU-only mode.")
    print("=" * 70)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("OpenCV", test_opencv),
        ("YOLOv8s (CPU)", test_yolo_cpu),
        ("MobileNetV3 (CPU)", test_mobilenet_cpu)
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    print("\n" + "=" * 70)
    print("Test Results Summary:")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:25} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("✅ All tests passed! System is ready for CPU-only operation.")
        print("\nNext steps:")
        print("1. Build the ROS2 package: colcon build --packages-select drone_object_recognition")
        print("2. Run the system (CPU-only): ros2 launch drone_object_recognition object_recognition_cpu.launch.py")
        print("3. Test with USB camera (CPU-only): ros2 launch drone_object_recognition test_with_usb_camera_cpu.launch.py")
        
        print("\n✅ CPU-only mode is working correctly!")
        print("   All models will run on CPU for consistent performance.")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\nCommon solutions:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Check internet connection for model downloads")
        print("- Ensure all Python packages are properly installed")
    
    return all_passed


if __name__ == "__main__":
    main()
