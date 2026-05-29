# camera.py
# This file handles ONLY the webcam connection and frame reading.
# Think of this as the "camera driver" of our system.

import cv2                  # OpenCV library for computer vision
import sys                  # Used to exit the program cleanly
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

def open_camera():
    """
    Opens the webcam and returns the VideoCapture object.
    VideoCapture is OpenCV's way of connecting to a camera.
    """
    # cv2.VideoCapture(0) means: connect to camera at index 0
    # Index 0 = your first/default webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check if it is connected.")
        sys.exit(1)   # Exit program with error code 1

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print(f"[INFO] Camera opened successfully at index {CAMERA_INDEX}")
    print(f"[INFO] Resolution set to {FRAME_WIDTH}x{FRAME_HEIGHT}")

    return cap


def release_camera(cap):
    """
    Safely releases the camera and closes all OpenCV windows.
    Always call this when you are done using the camera.
    """
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released.")


# ---- MAIN TEST BLOCK ----
# This code ONLY runs if you run camera.py directly.
# It does NOT run if another file imports camera.py.
if __name__ == "__main__":

    print("[INFO] Starting webcam test...")
    cap = open_camera()

    while True:
        # cap.read() reads ONE frame from the webcam
        # ret = True if frame was read successfully, False if not
        # frame = the actual image (a NumPy array of pixels)
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        # Display the frame in a window called "Webcam Feed"
        cv2.imshow("Webcam Feed", frame)

        # Wait 1ms for a key press.
        # If the key pressed is 'q', break the loop and stop.
        # 0xFF is a bitmask — required on some systems for correct key detection
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] 'q' pressed. Exiting...")
            break

    # Always release resources when loop ends
    release_camera(cap)