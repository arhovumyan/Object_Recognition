#!/usr/bin/env python3
"""Quick camera test to verify camera access."""

import cv2

print("Testing camera access...")
cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("✅ Camera 0 is accessible!")
    ret, frame = cap.read()
    if ret:
        print(f"✅ Successfully captured frame: {frame.shape}")
    else:
        print("❌ Could not capture frame")
    cap.release()
else:
    print("❌ Camera 0 is NOT accessible")

print("\nDone!")
