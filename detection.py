# detection.py
# This file handles ALL AI detection logic using YOLOv8.
# It loads the model once, then runs detection on every frame.
# Think of this as the "AI brain" of our surveillance system.

from ultralytics import YOLO   # Ultralytics library that runs YOLOv8
import cv2                     # OpenCV for drawing boxes on frames
from config import (
    YOLO_MODEL,
    CONFIDENCE_THRESHOLD,
    PERSON_CLASS_ID
)


def load_model():
    """
    Loads the YOLOv8 model from disk into memory.
    Call this ONCE at the start of your program.
    Loading takes 1-3 seconds — you do NOT want to load it every frame.
    """
    print(f"[INFO] Loading YOLO model from: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)   # loads the .pt file
    print("[INFO] YOLO model loaded successfully.")
    return model


def detect_persons(model, frame):
    """
    Runs YOLO detection on a single frame.
    Returns a list of detections, each containing:
        - bbox: (x1, y1, x2, y2) — bounding box corners
        - confidence: float like 0.87 — how sure YOLO is
        - label: string like "person"

    Parameters:
        model  — the loaded YOLO model object
        frame  — one image frame from the webcam (NumPy array)
    """

    # model(frame) runs inference — YOLO analyses the entire frame
    # verbose=False stops YOLO from printing logs every frame
    results = model(frame, verbose=False)

    # This list will hold all detected persons in this frame
    detections = []

    # results is a list — one result per image sent
    # We only sent one frame, so results[0] is our result
    result = results[0]

    # result.boxes contains all detected bounding boxes
    # Each box has: xyxy (coordinates), conf (confidence), cls (class id)
    for box in result.boxes:

        # box.cls is a tensor — .item() converts it to a plain Python number
        class_id = int(box.cls.item())

        # We only care about persons (class 0 in YOLO's COCO dataset)
        if class_id != PERSON_CLASS_ID:
            continue   # skip this detection, it's not a person

        # box.conf is confidence score — also a tensor
        confidence = float(box.conf.item())

        # Skip if confidence is below our threshold (e.g. below 0.5 = 50%)
        if confidence < CONFIDENCE_THRESHOLD:
            continue   # skip low-confidence detections

        # box.xyxy gives [[x1, y1, x2, y2]] as a tensor
        # [0] gets the first (only) row, .tolist() converts to Python list
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Build a clean detection dictionary
        detection = {
            "bbox": (x1, y1, x2, y2),
            "confidence": confidence,
            "label": "person"
        }

        detections.append(detection)

    return detections


def draw_detections(frame, detections):
    """
    Draws bounding boxes and labels on the frame for each detection.
    This modifies the frame in-place (directly changes the image).

    Parameters:
        frame      — webcam frame image (NumPy array)
        detections — list of detection dicts from detect_persons()
    """

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        confidence = det["confidence"]
        label = det["label"]

        # Draw a green rectangle around the detected person
        # Arguments: image, top-left corner, bottom-right corner, color (BGR), thickness
        # Note: OpenCV uses BGR color format, NOT RGB
        # (0, 255, 0) = Green in BGR
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Build the text label: "person 87%"
        text = f"{label} {confidence * 100:.0f}%"

        # Draw a filled black rectangle behind the text (makes it readable)
        # cv2.getTextSize returns the width and height of the text
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # Draw filled rectangle as text background
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 10),   # top-left of background box
            (x1 + text_width, y1),           # bottom-right of background box
            (0, 255, 0),                     # green background
            -1                               # -1 means filled (not just outline)
        )

        # Draw the label text on top of the green background
        # Arguments: image, text, position, font, scale, color, thickness
        cv2.putText(
            frame,
            text,
            (x1, y1 - 5),                  # slightly above the box
            cv2.FONT_HERSHEY_SIMPLEX,       # font style
            0.6,                            # font size scale
            (0, 0, 0),                      # black text color (BGR)
            2                               # thickness
        )

    return frame


def get_person_count(detections):
    """
    Returns how many persons are detected in the current frame.
    Simple helper used by the alert system later.
    """
    return len(detections)


# ---- MAIN TEST BLOCK ----
# Run this file directly to test YOLO detection on your webcam.
if __name__ == "__main__":

    # Import camera functions from camera.py
    from camera import open_camera, release_camera

    print("[INFO] Starting YOLO detection test...")

    # Load YOLO model ONCE before the loop
    model = load_model()

    # Open webcam
    cap = open_camera()

    while True:
        # Read one frame from webcam
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        # Run YOLO detection on the frame
        detections = detect_persons(model, frame)

        # Draw boxes on the frame
        frame = draw_detections(frame, detections)

        # Show person count on the top-left of the frame
        count = get_person_count(detections)
        count_text = f"Persons Detected: {count}"

        # Draw black background for counter text
        cv2.rectangle(frame, (5, 5), (250, 35), (0, 0, 0), -1)

        # Draw counter text in white
        cv2.putText(
            frame,
            count_text,
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),   # white text
            2
        )

        # Display the frame with detections
        cv2.imshow("YOLO Human Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting detection test...")
            break

    release_camera(cap)