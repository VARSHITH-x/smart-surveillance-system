# tests/test_combined.py
# Tests motion detection AND YOLO person detection running together.
# This is the core pipeline of the surveillance system.
#
# Logic:
#   Every frame  → check for motion (cheap, fast)
#   Motion found → run YOLO (expensive, accurate)
#   Both confirm → this is a real intrusion event

import sys
import os

# Add parent folder to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from camera import open_camera, release_camera
from detection import load_model, detect_persons, draw_detections
from motion import MotionDetector

print("[INFO] Starting combined motion + YOLO test...")

# Load YOLO model once before the loop
model = load_model()

# Create motion detector
motion_detector = MotionDetector()

# Open webcam
cap = open_camera()

# Frame counter for skipping YOLO on some frames
frame_count = 0
last_detections = []

print("[INFO] Press 'q' to quit.")
print("[INFO] Blue box = Motion region")
print("[INFO] Green box = YOLO person detection")

while True:
    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Failed to read frame.")
        break

    frame_count += 1

    # --- Step 1: Always run motion detection (fast) ---
    motion_found, motion_frame, contours = motion_detector.detect(frame)

    # --- Step 2: Only run YOLO when motion is detected (saves CPU) ---
    # Also only run YOLO every 3rd frame to reduce lag
    if motion_found and frame_count % 3 == 0:
        last_detections = detect_persons(model, frame)

    # If no motion, clear old detections
    if not motion_found:
        last_detections = []

    # --- Step 3: Draw YOLO detections on top of motion frame ---
    output_frame = draw_detections(motion_frame, last_detections)

    # --- Step 4: Build status display ---
    person_count = len(last_detections)

    # Show alert if BOTH motion AND person detected
    if motion_found and person_count > 0:
        alert_text = "!! INTRUSION ALERT !!"
        alert_color = (0, 0, 255)    # Red

        # Draw large red alert banner at bottom of frame
        h, w = output_frame.shape[:2]
        cv2.rectangle(output_frame, (0, h - 50), (w, h), (0, 0, 200), -1)
        cv2.putText(
            output_frame,
            alert_text,
            (w // 2 - 150, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )
    elif motion_found:
        # Motion but no person confirmed yet
        cv2.putText(
            output_frame,
            "Motion - Checking...",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),   # Orange
            2
        )

    # Show person count
    cv2.rectangle(output_frame, (5, 40), (270, 70), (0, 0, 0), -1)
    cv2.putText(
        output_frame,
        f"Persons: {person_count}",
        (10, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # Display final output frame
    cv2.imshow("Surveillance Feed - Combined Test", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

release_camera(cap)