# app.py — Safe version with full error logging

import cv2
import threading
import time
import traceback
from flask import Flask, Response, render_template, jsonify
from datetime import datetime

print("[STARTUP] Importing modules...")

try:
    from camera    import open_camera, release_camera
    print("[OK] camera")
    from detection import load_model, detect_persons, draw_detections
    print("[OK] detection")
    from motion    import MotionDetector
    print("[OK] motion")
    from alerts    import (IntrusionDetector,
                           STATE_IDLE, STATE_DETECTING,
                           STATE_ALERT, STATE_COOLDOWN)
    print("[OK] alerts")
    from database  import get_recent_alerts, get_alert_count, get_todays_alerts
    print("[OK] database")
    from config    import (FLASK_HOST, FLASK_PORT, FRAME_WIDTH,
                           FRAME_HEIGHT, INTRUSION_TRIGGER_SECONDS)
    print("[OK] config")
except Exception as e:
    print(f"\n[IMPORT ERROR] {e}")
    traceback.print_exc()
    exit(1)

print("[STARTUP] All imports successful.\n")

# ── Flask ─────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='static')

# ── Shared globals ────────────────────────────────────────────────
latest_frame     = None
frame_lock       = threading.Lock()
system_state     = "IDLE"
person_count     = 0
alert_count_live = 0
is_recording     = False
thread_error     = None   # stores any crash from camera thread


# ── Surveillance thread ───────────────────────────────────────────

def surveillance_thread():
    global latest_frame, system_state, person_count
    global alert_count_live, is_recording, thread_error

    try:
        print("[THREAD] Opening camera...")
        cap = open_camera()
        print("[THREAD] Loading YOLO model...")
        model     = load_model()
        motion    = MotionDetector()
        intrusion = IntrusionDetector()
        print("[THREAD] All components ready. Starting loop...")

        frame_count = 0
        last_det    = []

        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_count += 1

            motion_found, mframe, _ = motion.detect(frame)

            if motion_found and frame_count % 3 == 0:
                last_det = detect_persons(model, frame)
            if not motion_found:
                last_det = []

            pc = len(last_det)

            _, state, tim = intrusion.update(
                motion_detected = motion_found,
                person_detected = pc > 0,
                person_count    = pc,
                frame           = mframe
            )

            out = draw_detections(mframe, last_det)
            out = _draw_hud(out, state, pc,
                            intrusion.alert_count,
                            intrusion.recorder.is_recording, tim)

            with frame_lock:
                latest_frame     = out.copy()
                system_state     = state
                person_count     = pc
                alert_count_live = intrusion.alert_count
                is_recording     = intrusion.recorder.is_recording

    except Exception as e:
        thread_error = str(e)
        print(f"\n[THREAD CRASH] {e}")
        traceback.print_exc()


def _draw_hud(frame, state, persons, alerts, recording, tim):
    h, w = frame.shape[:2]
    colors = {
        "IDLE":      (0, 200, 0),
        "DETECTING": (0, 165, 255),
        "ALERT":     (0, 0, 255),
        "COOLDOWN":  (180, 0, 180),
    }
    sc = colors.get(state, (255, 255, 255))
    cv2.rectangle(frame, (0, 0), (w, 65), (0, 0, 0), -1)
    cv2.putText(frame, f"STATE: {state}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc, 2)
    cv2.putText(frame, f"Persons:{persons}  Alerts:{alerts}",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200,200,200), 1)
    cv2.putText(frame, datetime.now().strftime("%H:%M:%S"),
                (w-85, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130,130,130), 1)
    if state == "DETECTING" and tim > 0:
        p = min(tim / INTRUSION_TRIGGER_SECONDS, 1.0)
        cv2.rectangle(frame,(10,55),(w-10,63),(40,40,40),-1)
        cv2.rectangle(frame,(10,55),(10+int((w-20)*p),63),(0,165,255),-1)
    if recording:
        cv2.circle(frame, (w-22, 48), 9, (0,0,255), -1)
    return frame


# ── Stream generator ──────────────────────────────────────────────

def generate_frames():
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            # Send a black placeholder frame while camera warms up
            import numpy as np
            placeholder = np.zeros((480, 640, 3), dtype='uint8')
            cv2.putText(placeholder, "Camera initializing...",
                        (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,100,100), 2)
            ret, buf = cv2.imencode('.jpg', placeholder)
        else:
            ret, buf = cv2.imencode('.jpg', frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, 70])

        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')
        time.sleep(0.05)


# ── Routes ────────────────────────────────────────────────────────

@app.route('/')
def index():
    try:
        alerts        = get_recent_alerts(15)
        total_alerts  = get_alert_count()
        todays        = len(get_todays_alerts())
    except Exception as e:
        print(f"[DB ERROR] {e}")
        alerts = []
        total_alerts = 0
        todays = 0

    return render_template('index.html',
        system_state    = system_state,
        person_count    = person_count,
        alert_count     = alert_count_live,
        is_recording    = is_recording,
        total_db_alerts = total_alerts,
        todays_alerts   = todays,
        alerts          = alerts,
        thread_error    = thread_error,
    )


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/status')
def api_status():
    return jsonify({
        "state":        system_state,
        "person_count": person_count,
        "alert_count":  alert_count_live,
        "is_recording": is_recording,
        "total_alerts": get_alert_count(),
        "thread_error": thread_error,
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


@app.route('/api/alerts')
def api_alerts():
    return jsonify(get_recent_alerts(20))


# ── Start ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 50)
    print("  Smart Surveillance System")
    print("=" * 50)

    t = threading.Thread(
        target=surveillance_thread,
        daemon=True,
        name="SurveillanceThread"
    )
    t.start()
    print("[MAIN] Surveillance thread launched.")
    print("[MAIN] Waiting 3s for camera warmup...")
    time.sleep(3)

    print(f"\n[MAIN] Open browser → http://localhost:{FLASK_PORT}\n")

    app.run(
        host        = '0.0.0.0',
        port        = FLASK_PORT,
        debug       = False,
        threaded    = True,
        use_reloader= False
    )