# motion.py
# Handles all motion detection logic using frame differencing.
# This is a classical computer vision technique — no AI needed here.
# Works fast on CPU because it's just pixel math, not neural networks.

import cv2      # OpenCV for image processing
import numpy as np  # NumPy for array operations
from config import MOTION_THRESHOLD


class MotionDetector:
    """
    A class that detects motion between consecutive webcam frames.

    Why a class?
    Because we need to REMEMBER the previous frame between calls.
    A plain function forgets everything after it returns.
    A class stores data (previous frame) across multiple calls.
    """

    def __init__(self):
        """
        __init__ runs once when you create a MotionDetector object.
        Sets up initial state.
        """
        # This stores the previous frame for comparison
        # Starts as None because there's no previous frame yet
        self.previous_frame = None

        # Tracks if motion is currently happening
        self.motion_detected = False

        # Stores the contours (motion regions) found in last detection
        self.motion_contours = []

        print("[INFO] MotionDetector initialized.")

    def preprocess_frame(self, frame):
        """
        Converts a raw webcam frame into a clean grayscale image
        ready for motion comparison.

        Steps:
        1. Convert to grayscale  (remove color, reduce data)
        2. Apply Gaussian blur   (remove noise)

        Parameters:
            frame — raw BGR webcam frame (NumPy array H x W x 3)

        Returns:
            processed — grayscale blurred frame (NumPy array H x W)
        """

        # Step 1: Convert from BGR (color) to GRAY (grayscale)
        # cv2.COLOR_BGR2GRAY tells OpenCV the conversion direction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 2: Apply Gaussian Blur to reduce sensor noise
        # (21, 21) = kernel size — must be ODD numbers
        # Larger kernel = more blur = less sensitive to tiny changes
        # 0 = let OpenCV calculate sigma (blur strength) automatically
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        return blurred

    def detect(self, frame):
        """
        Main method — call this every frame.
        Compares current frame with previous frame to detect motion.

        Parameters:
            frame — current webcam frame (BGR NumPy array)

        Returns:
            motion_detected — True if motion found, False if not
            motion_frame    — frame with motion regions drawn on it
            contours        — list of motion contour objects
        """

        # Preprocess the current frame
        processed = self.preprocess_frame(frame)

        # Make a copy of the frame to draw on
        # We draw on a copy so the original stays clean
        motion_frame = frame.copy()

        # --- FIRST FRAME CASE ---
        # If this is the very first frame, there's nothing to compare
        # Save it and return False (no motion yet)
        if self.previous_frame is None:
            self.previous_frame = processed
            return False, motion_frame, []

        # --- FRAME DIFFERENCING ---
        # cv2.absdiff = absolute difference between two images
        # For each pixel: result = |prev_pixel - curr_pixel|
        # If pixel didn't change: result ≈ 0  (dark)
        # If pixel changed a lot: result ≈ 255 (bright)
        frame_diff = cv2.absdiff(self.previous_frame, processed)

        # --- THRESHOLDING ---
        # Convert the difference image to pure black & white
        # Pixels with difference > 25 → 255 (white = motion)
        # Pixels with difference ≤ 25 → 0   (black = no motion)
        # cv2.THRESH_BINARY = simple binary threshold type
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        # --- DILATION ---
        # Dilation expands the white regions slightly
        # This fills in small gaps between nearby motion pixels
        # making the motion blob more solid and easier to detect
        # np.ones((3,3)) = 3x3 kernel of all ones
        # iterations=2 = apply dilation twice
        dilated = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=2)

        # --- CONTOUR DETECTION ---
        # Find the outlines of all white blobs in the threshold image
        # cv2.RETR_EXTERNAL = only find outer contours (not holes inside)
        # cv2.CHAIN_APPROX_SIMPLE = compress contour points (saves memory)
        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # --- FILTER SMALL CONTOURS ---
        # Not all contours are real motion — filter by area size
        motion_found = False
        valid_contours = []

        for contour in contours:
            # Calculate the area of this contour in pixels
            area = cv2.contourArea(contour)

            # Ignore small blobs — they are noise or tiny movements
            # MOTION_THRESHOLD = 5000 pixels (set in config.py)
            if area < MOTION_THRESHOLD:
                continue  # skip this small contour

            # This contour is large enough to be real motion
            motion_found = True
            valid_contours.append(contour)

            # Get bounding rectangle around the motion region
            # cv2.boundingRect returns (x, y, width, height)
            x, y, w, h = cv2.boundingRect(contour)

            # Draw a BLUE rectangle around the motion area on the frame
            # (255, 0, 0) = Blue in BGR format
            cv2.rectangle(
                motion_frame,
                (x, y),          # top-left corner
                (x + w, y + h),  # bottom-right corner
                (255, 0, 0),     # blue color
                2                # line thickness
            )

        # --- DRAW STATUS TEXT ---
        # Show whether motion is detected on the frame
        if motion_found:
            status_text = "MOTION DETECTED"
            status_color = (0, 0, 255)    # Red in BGR
        else:
            status_text = "No Motion"
            status_color = (0, 255, 0)    # Green in BGR

        # Draw black background box for the status text
        cv2.rectangle(motion_frame, (5, 5), (270, 35), (0, 0, 0), -1)

        # Draw the status text
        cv2.putText(
            motion_frame,
            status_text,
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )

        # --- UPDATE STATE ---
        # Save current frame as previous for the NEXT iteration
        self.previous_frame = processed

        # Save state for external access
        self.motion_detected = motion_found
        self.motion_contours = valid_contours

        return motion_found, motion_frame, valid_contours

    def reset(self):
        """
        Resets the detector — clears the stored previous frame.
        Use this if the camera restarts or pauses.
        """
        self.previous_frame = None
        self.motion_detected = False
        self.motion_contours = []
        print("[INFO] MotionDetector reset.")

    def get_motion_area(self):
        """
        Returns the total pixel area of all motion regions combined.
        Useful for measuring HOW MUCH motion is happening.
        """
        total_area = 0
        for contour in self.motion_contours:
            total_area += cv2.contourArea(contour)
        return total_area


# ---- MAIN TEST BLOCK ----
# Run this file directly to test motion detection on your webcam.
if __name__ == "__main__":

    from camera import open_camera, release_camera

    print("[INFO] Starting motion detection test...")
    print("[INFO] Move in front of webcam to trigger motion detection.")
    print("[INFO] Press 'q' to quit.")

    # Create motion detector object
    detector = MotionDetector()

    # Open webcam
    cap = open_camera()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        # Run motion detection on the current frame
        motion_found, motion_frame, contours = detector.detect(frame)

        # Print to terminal when motion state changes
        if motion_found:
            area = detector.get_motion_area()
            print(f"[MOTION] Detected! Total motion area: {area:.0f} px²")

        # Show the frame with motion boxes drawn
        cv2.imshow("Motion Detection", motion_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting motion detection test...")
            break

    release_camera(cap)