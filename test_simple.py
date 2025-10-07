#!/usr/bin/env python3

"""
Simple test script to verify the object recognition system components.
This script tests the YOLO and MobileNet models without ROS2.

Note: Make sure to activate the virtual environment first:
source venv/bin/activate
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


def test_yolo():
    """Test YOLOv8s model loading and basic functionality."""
    print("🔍 Testing YOLOv8s...")
    
    try:
        # Load model
        model = YOLO('yolov8s.pt')
        print("✅ YOLOv8s model loaded successfully")
        
        # Create a test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run inference
        results = model(test_image, conf=0.5)
        print("✅ YOLO inference completed")
        
        # Print available classes
        print(f"📋 Available YOLO classes: {len(model.names)} classes")
        target_classes = ['person', 'cell phone', 'laptop', 'tv']
        for cls in target_classes:
            if cls in model.names.values():
                class_id = list(model.names.keys())[list(model.names.values()).index(cls)]
                print(f"   - {cls}: class_id {class_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO test failed: {str(e)}")
        return False


def test_mobilenet():
    """Test MobileNetV3 model loading and basic functionality."""
    print("\n🧠 Testing MobileNetV3...")
    
    try:
        # Load model
        model = MobileNetV3Small(
            input_shape=(224, 224, 3),
            weights='imagenet',
            include_top=True
        )
        print("✅ MobileNetV3 model loaded successfully")
        
        # Create a test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Preprocess
        img_array = image.img_to_array(test_image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Run inference
        predictions = model.predict(img_array, verbose=0)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        
        print("✅ MobileNetV3 inference completed")
        print("📋 Sample predictions:")
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
            print(f"🎯 Found potential target objects: {found_targets}")
        else:
            print("ℹ️  No direct target objects in top predictions (this is normal for random images)")
        
        return True
        
    except Exception as e:
        print(f"❌ MobileNetV3 test failed: {str(e)}")
        return False


def test_opencv():
    """Test OpenCV functionality."""
    print("\n📷 Testing OpenCV...")
    
    try:
        # Test basic OpenCV functionality
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), 2)
        
        # Test image operations
        resized = cv2.resize(test_image, (224, 224))
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        print("✅ OpenCV basic operations working")
        print(f"📏 Image shape: {test_image.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenCV test failed: {str(e)}")
        return False


def test_dependencies():
    """Test all required dependencies."""
    print("\n📦 Testing dependencies...")
    
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
    """Run all tests."""
    print("🚁 Drone Object Recognition System - Component Tests")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("OpenCV", test_opencv),
        ("YOLOv8s", test_yolo),
        ("MobileNetV3", test_mobilenet)
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 All tests passed! System is ready for ROS2 integration.")
        print("\nNext steps:")
        print("1. Build the ROS2 package: colcon build --packages-select drone_object_recognition")
        print("2. Run the system: ros2 launch drone_object_recognition object_recognition.launch.py")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("\nCommon solutions:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Check CUDA installation if using GPU")
        print("- Ensure internet connection for model downloads")
    
    return all_passed


if __name__ == "__main__":
    main()
