# app.py
# Entry point — starts threading_manager, runs Flask.
# This file now ONLY handles web routes.
# All thread logic lives in threading_manager.py

import cv2
import time
import numpy as np
from flask import Flask, Response, render_template, jsonify
from datetime import datetime

from threading_manager import (
    start_surveillance,
    stop_surveillance,
    get_latest_frame,
    get_system_status,
    is_thread_running
)
from database import get_recent_alerts, get_alert_count, get_todays_alerts
from config   import FLASK_HOST, FLASK_PORT


app = Flask(__name__, static_folder='static')


# ── MJPEG stream generator ────────────────────────────────────────

def generate_frames():
    while True:
        frame = get_latest_frame()

        if frame is None:
            # Black placeholder while camera warms up
            placeholder = np.zeros((480, 640, 3), dtype='uint8')
            cv2.putText(placeholder, "Camera initializing...",
                        (140, 240), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (80, 80, 80), 2)
            ret, buf = cv2.imencode('.jpg', placeholder)
        else:
            ret, buf = cv2.imencode(
                '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )

        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')

        time.sleep(0.05)   # ~20fps cap


# ── Routes ────────────────────────────────────────────────────────

@app.route('/')
def index():
    status = get_system_status()
    return render_template('index.html',
        system_state    = status["state"],
        person_count    = status["person_count"],
        alert_count     = status["alert_count"],
        is_recording    = status["is_recording"],
        total_db_alerts = get_alert_count(),
        todays_alerts   = len(get_todays_alerts()),
        alerts          = get_recent_alerts(15),
    )


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/status')
def api_status():
    status = get_system_status()
    return jsonify({
        **status,
        "total_alerts":  get_alert_count(),
        "todays_alerts": len(get_todays_alerts()),
        "thread_alive":  is_thread_running(),
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


@app.route('/api/alerts')
def api_alerts():
    return jsonify(get_recent_alerts(20))


# ── Entry point ───────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 50)
    print("  Smart Surveillance System")
    print("=" * 50)

    # Start surveillance pipeline (threading_manager handles it)
    start_surveillance()

    print("[MAIN] Waiting 3s for camera warmup...")
    time.sleep(3)

    print(f"\n[MAIN] Dashboard → http://localhost:{FLASK_PORT}\n")

    try:
        app.run(
            host        = '0.0.0.0',
            port        = FLASK_PORT,
            debug       = False,
            threaded    = True,
            use_reloader= False
        )
    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down...")
        stop_surveillance()
        print("[MAIN] Done.")