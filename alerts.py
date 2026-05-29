# alerts.py
import time
import os
from datetime import datetime
import cv2

from config import (
    INTRUSION_TRIGGER_SECONDS,
    ALERT_COOLDOWN_SECONDS,
    SCREENSHOT_FOLDER,
    SCREENSHOT_PREFIX
)
from database import (
    initialize_database, log_alert,
    log_system_event, update_daily_stats
)
from recorder import VideoRecorder

STATE_IDLE      = "IDLE"
STATE_DETECTING = "DETECTING"
STATE_ALERT     = "ALERT"
STATE_COOLDOWN  = "COOLDOWN"


class IntrusionDetector:

    def __init__(self):
        self.state               = STATE_IDLE
        self.motion_start_time   = None
        self.last_alert_time     = None
        self.alert_count         = 0
        self.alert_history       = []

        initialize_database()
        log_system_event("INFO", "Surveillance system started")

        self.recorder = VideoRecorder()
        os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)

        print("[INFO] IntrusionDetector ready.")
        print(f"[INFO] Trigger : {INTRUSION_TRIGGER_SECONDS}s | Cooldown: {ALERT_COOLDOWN_SECONDS}s")

    def update(self, motion_detected, person_detected, person_count, frame):
        current_time   = time.time()
        alert_fired    = False
        time_in_motion = 0.0

        # Always feed frames to recorder when active
        if self.recorder.is_recording:
            self.recorder.add_frame(frame)

        # ── IDLE ─────────────────────────────────────────────────
        if self.state == STATE_IDLE:
            if motion_detected and person_detected:
                self.state             = STATE_DETECTING
                self.motion_start_time = current_time
                print("[DETECTING] Started timing...")

        # ── DETECTING ────────────────────────────────────────────
        elif self.state == STATE_DETECTING:
            if not motion_detected or not person_detected:
                elapsed = current_time - self.motion_start_time
                print(f"[IDLE] Reset at {elapsed:.1f}s (needed {INTRUSION_TRIGGER_SECONDS}s)")
                self.state             = STATE_IDLE
                self.motion_start_time = None
            else:
                elapsed        = current_time - self.motion_start_time
                time_in_motion = elapsed
                print(f"[DETECTING] {elapsed:.1f}s / {INTRUSION_TRIGGER_SECONDS}s", end="\r")

                if elapsed >= INTRUSION_TRIGGER_SECONDS:
                    self.state  = STATE_ALERT
                    self._fire_alert(frame, person_count, current_time)
                    alert_fired = True

        # ── ALERT ────────────────────────────────────────────────
        elif self.state == STATE_ALERT:
            self.state           = STATE_COOLDOWN
            self.last_alert_time = current_time
            print(f"[COOLDOWN] {ALERT_COOLDOWN_SECONDS}s cooldown started.")

        # ── COOLDOWN ─────────────────────────────────────────────
        elif self.state == STATE_COOLDOWN:
            if current_time - self.last_alert_time >= ALERT_COOLDOWN_SECONDS:
                self.state             = STATE_IDLE
                self.motion_start_time = None
                print("[IDLE] Cooldown done. Ready.")

        return alert_fired, self.state, time_in_motion

    def _fire_alert(self, frame, person_count, current_time):
        self.alert_count  += 1
        ts_file            = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ts_readable        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Screenshot
        ss_filename   = f"{SCREENSHOT_PREFIX}_{ts_file}.jpg"
        ss_path       = os.path.join(SCREENSHOT_FOLDER, ss_filename)
        _save_screenshot(frame, ss_path, self.alert_count, person_count)

        # 2. Start video
        video_path = self.recorder.start_recording(alert_id=self.alert_count)

        # 3. Database
        db_id = log_alert(
            timestamp        = ts_readable,
            person_count     = person_count,
            screenshot_path  = ss_path,
            video_path       = video_path,
            duration_seconds = 0,
            notes            = f"Alert #{self.alert_count}"
        )
        update_daily_stats(person_count)
        log_system_event("WARNING",
            f"Alert #{self.alert_count} — {person_count} person(s)")

        self.alert_history.append({
            "id":              self.alert_count,
            "db_id":           db_id,
            "timestamp":       ts_readable,
            "timestamp_unix":  current_time,
            "person_count":    person_count,
            "screenshot_path": ss_path,
            "screenshot_file": ss_filename,
            "video_path":      video_path,
        })

        print(f"\n{'='*50}")
        print(f"  ALERT #{self.alert_count} FIRED")
        print(f"  Time       : {ts_readable}")
        print(f"  Persons    : {person_count}")
        print(f"  Screenshot : {ss_path}")
        print(f"  Video      : {video_path}")
        print(f"  DB ID      : {db_id}")
        print(f"{'='*50}\n")

    def get_status(self):
        rec = self.recorder.get_status()
        return {
            "state":         self.state,
            "alert_count":   self.alert_count,
            "is_recording":  rec["is_recording"],
            "recorder":      rec,
            "last_alert":    self.alert_history[-1] if self.alert_history else None,
            "alert_history": self.alert_history[-10:]
        }

    def shutdown(self):
        if self.recorder.is_recording:
            self.recorder.stop_recording()
        log_system_event("INFO", "Surveillance system stopped")
        print("[INFO] Shutdown complete.")


def _save_screenshot(frame, filepath, alert_id, person_count):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        img       = frame.copy()
        h, w      = img.shape[:2]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cv2.rectangle(img, (0, h - 35), (w, h), (0, 0, 0), -1)
        cv2.putText(img,
                    f"ALERT #{alert_id}  |  {timestamp}  |  Persons: {person_count}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        ok = cv2.imwrite(filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            print(f"[SCREENSHOT] Saved → {filepath}")
        else:
            print(f"[SCREENSHOT ERROR] imwrite failed: {filepath}")
        return ok
    except Exception as e:
        print(f"[SCREENSHOT ERROR] {e}")
        return False


# ── TEST ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from camera import open_camera, release_camera
    from detection import load_model, detect_persons, draw_detections
    from motion import MotionDetector

    print("Move for 2s to trigger alert. Press 'q' to quit.\n")

    model    = load_model()
    motion   = MotionDetector()
    detector = IntrusionDetector()
    cap      = open_camera()
    count    = 0
    last_det = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

        motion_found, mframe, _ = motion.detect(frame)

        if motion_found and count % 3 == 0:
            last_det = detect_persons(model, frame)
        if not motion_found:
            last_det = []

        pc = len(last_det)
        alert_fired, state, tim = detector.update(
            motion_detected = motion_found,
            person_detected = pc > 0,
            person_count    = pc,
            frame           = mframe
        )

        out = draw_detections(mframe, last_det)
        h, w = out.shape[:2]

        colors = {"IDLE":(0,200,0),"DETECTING":(0,165,255),
                  "ALERT":(0,0,255),"COOLDOWN":(200,0,200)}
        cv2.rectangle(out, (0,0), (w,60), (0,0,0), -1)
        cv2.putText(out, f"STATE: {state}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors.get(state,(255,255,255)), 2)
        cv2.putText(out, f"Persons:{pc}  Alerts:{detector.alert_count}",
                    (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

        if state == "DETECTING" and tim > 0:
            p = min(tim / INTRUSION_TRIGGER_SECONDS, 1.0)
            cv2.rectangle(out,(10,55),(w-10,65),(40,40,40),-1)
            cv2.rectangle(out,(10,55),(10+int((w-20)*p),65),(0,165,255),-1)

        if detector.recorder.is_recording:
            cv2.circle(out, (w-25,25), 10, (0,0,255), -1)
            cv2.putText(out,"REC",(w-70,32),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,255),2)

        cv2.imshow("Surveillance Test", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            detector.shutdown()
            break

    release_camera(cap)
    print(f"\nDone. Alerts: {detector.alert_count}")
    print(f"Screenshots : {SCREENSHOT_FOLDER}/")
    print(f"Videos      : static/videos/")
    print(f"Database    : logs/surveillance.db")