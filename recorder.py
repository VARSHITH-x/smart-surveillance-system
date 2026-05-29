# recorder.py
import cv2
import os
import time
from datetime import datetime
from config import VIDEO_FOLDER, CLIP_DURATION_SECONDS, FRAME_WIDTH, FRAME_HEIGHT, FPS


class VideoRecorder:

    def __init__(self):
        self.writer               = None
        self.is_recording         = False
        self.current_filepath     = None
        self.recording_start_time = None
        self.frames_written       = 0
        self._codec               = None
        os.makedirs(VIDEO_FOLDER, exist_ok=True)
        self.fourcc, self.ext = self._find_working_codec()
        print(f"[RECORDER] Ready. Codec: {self._codec}  Extension: {self.ext}")

    def _find_working_codec(self):
        """Tests codecs and returns the first one that works."""
        test_path = os.path.join(VIDEO_FOLDER, "_test.mp4")
        for codec, ext in [('mp4v', '.mp4'), ('XVID', '.avi'), ('MJPG', '.avi')]:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test   = cv2.VideoWriter(
                    test_path, fourcc, float(FPS), (FRAME_WIDTH, FRAME_HEIGHT)
                )
                if test.isOpened():
                    test.release()
                    if os.path.exists(test_path):
                        os.remove(test_path)
                    self._codec = codec
                    return fourcc, ext
                test.release()
            except Exception:
                continue
        # Hard fallback
        self._codec = 'mp4v'
        return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'

    def start_recording(self, alert_id=None):
        if self.is_recording:
            self.stop_recording()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename  = (f"clip_alert{alert_id}_{timestamp}{self.ext}"
                     if alert_id else f"clip_{timestamp}{self.ext}")
        filepath  = os.path.join(VIDEO_FOLDER, filename)

        self.writer = cv2.VideoWriter(
            filepath,
            self.fourcc,
            float(FPS),
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        if not self.writer.isOpened():
            print(f"[RECORDER ERROR] Cannot open VideoWriter → {filepath}")
            self.writer = None
            return None

        self.is_recording         = True
        self.current_filepath     = filepath
        self.recording_start_time = time.time()
        self.frames_written       = 0
        print(f"[RECORDER] Recording started → {filepath}")
        return filepath

    def add_frame(self, frame):
        if not self.is_recording or self.writer is None:
            return False

        elapsed = time.time() - self.recording_start_time
        if elapsed >= CLIP_DURATION_SECONDS:
            print(f"[RECORDER] {CLIP_DURATION_SECONDS}s done. Auto-stopping.")
            self.stop_recording()
            return False

        h, w = frame.shape[:2]
        if w != FRAME_WIDTH or h != FRAME_HEIGHT:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        self.writer.write(frame)
        self.frames_written += 1
        return True

    def stop_recording(self):
        if not self.is_recording:
            return None

        filepath = self.current_filepath
        duration = round(time.time() - self.recording_start_time, 1) \
                   if self.recording_start_time else 0

        if self.writer:
            self.writer.release()
            self.writer = None

        self.is_recording         = False
        self.current_filepath     = None
        self.recording_start_time = None

        if filepath and os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"[RECORDER] Saved → {filepath}")
            print(f"[RECORDER] Size: {size} bytes | Duration: {duration}s | Frames: {self.frames_written}")
        else:
            print(f"[RECORDER ERROR] File missing after save: {filepath}")

        self.frames_written = 0
        return filepath

    def get_status(self):
        if self.is_recording and self.recording_start_time:
            elapsed   = round(time.time() - self.recording_start_time, 1)
            remaining = max(0.0, CLIP_DURATION_SECONDS - elapsed)
        else:
            elapsed = remaining = 0.0
        return {
            "is_recording":      self.is_recording,
            "current_file":      self.current_filepath,
            "frames_written":    self.frames_written,
            "elapsed_seconds":   elapsed,
            "remaining_seconds": round(remaining, 1)
        }


# ── TEST ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from camera import open_camera, release_camera

    print("'r' = record  |  's' = stop  |  'q' = quit")
    recorder = VideoRecorder()
    cap      = open_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if recorder.is_recording:
            recorder.add_frame(frame)

        display = frame.copy()
        st      = recorder.get_status()

        if st["is_recording"]:
            cv2.circle(display, (25, 25), 12, (0, 0, 255), -1)
            cv2.putText(display,
                        f"REC {st['elapsed_seconds']:.1f}s / {CLIP_DURATION_SECONDS}s",
                        (45, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        else:
            cv2.putText(display, "Press 'r' to start recording",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.imshow("Recorder Test", display)
        key = cv2.waitKey(1) & 0xFF

        if   key == ord('r') and not recorder.is_recording:
            recorder.start_recording()
        elif key == ord('s') and recorder.is_recording:
            recorder.stop_recording()
        elif key == ord('q'):
            if recorder.is_recording:
                recorder.stop_recording()
            break

    release_camera(cap)