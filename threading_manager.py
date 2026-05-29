# threading_manager.py
# Central thread manager for the surveillance system.
# Responsibilities:
#   - Hold ALL shared state between threads
#   - Start and stop the surveillance thread
#   - Provide thread-safe access to shared data
#   - Keep app.py clean (only Flask routes)

import threading
import time
import traceback
import cv2
import numpy as np
from datetime import datetime

from camera    import open_camera, release_camera
from detection import load_model, detect_persons, draw_detections
from motion    import MotionDetector
from alerts    import (IntrusionDetector,
                       STATE_IDLE, STATE_DETECTING,
                       STATE_ALERT, STATE_COOLDOWN)
from config    import FRAME_WIDTH, FRAME_HEIGHT, INTRUSION_TRIGGER_SECONDS


# ── Shared State ──────────────────────────────────────────────────
# These variables are READ by Flask, WRITTEN by camera thread.
# All access must go through the Lock.

_latest_frame     = None          # most recent processed frame
_frame_lock       = threading.Lock()

_system_state     = STATE_IDLE
_person_count     = 0
_alert_count      = 0
_is_recording     = False
_thread_error     = None          # stores crash message if thread dies

_surveillance_thread   = None     # thread object
_stop_event            = threading.Event()   # signal thread to stop


# ── Public Getters (used by app.py) ───────────────────────────────

def get_latest_frame():
    """Returns a copy of the latest processed frame. Thread-safe."""
    with _frame_lock:
        if _latest_frame is None:
            return None
        return _latest_frame.copy()


def get_system_status():
    """
    Returns all shared state as a dict.
    Flask calls this to build the dashboard response.
    """
    return {
        "state":        _system_state,
        "person_count": _person_count,
        "alert_count":  _alert_count,
        "is_recording": _is_recording,
        "thread_error": _thread_error,
    }


def is_thread_running():
    """Returns True if surveillance thread is alive."""
    return _surveillance_thread is not None and _surveillance_thread.is_alive()


# ── Surveillance Thread Function ──────────────────────────────────

def _surveillance_loop():
    """
    The main surveillance pipeline.
    Runs in a background thread.
    Writes results to shared global variables.
    """
    global _latest_frame, _system_state, _person_count
    global _alert_count, _is_recording, _thread_error

    try:
        print("[THREAD] Opening camera...")
        cap = open_camera()

        print("[THREAD] Loading YOLO model...")
        model     = load_model()
        motion    = MotionDetector()
        intrusion = IntrusionDetector()

        print("[THREAD] Pipeline ready. Starting loop...")

        frame_count = 0
        last_det    = []

        # Loop runs until stop_event is set (shutdown signal)
        while not _stop_event.is_set():

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_count += 1

            # ── Motion detection (every frame — fast) ─────────────
            motion_found, mframe, _ = motion.detect(frame)

            # ── YOLO (only on motion, every 3rd frame — saves CPU) ─
            if motion_found and frame_count % 3 == 0:
                last_det = detect_persons(model, frame)
            if not motion_found:
                last_det = []

            pc = len(last_det)

            # ── State machine ─────────────────────────────────────
            _, state, tim = intrusion.update(
                motion_detected = motion_found,
                person_detected = pc > 0,
                person_count    = pc,
                frame           = mframe
            )

            # ── Draw boxes + HUD ──────────────────────────────────
            out = draw_detections(mframe, last_det)
            out = _draw_hud(
                out, state, pc,
                intrusion.alert_count,
                intrusion.recorder.is_recording,
                tim
            )

            # ── Write to shared variables (thread-safe) ───────────
            with _frame_lock:
                _latest_frame  = out.copy()
                _system_state  = state
                _person_count  = pc
                _alert_count   = intrusion.alert_count
                _is_recording  = intrusion.recorder.is_recording

        # Loop ended — clean up
        print("[THREAD] Stop signal received. Cleaning up...")
        intrusion.shutdown()
        release_camera(cap)
        print("[THREAD] Surveillance thread stopped cleanly.")

    except Exception as e:
        _thread_error = str(e)
        print(f"\n[THREAD CRASH] {e}")
        traceback.print_exc()


def _draw_hud(frame, state, persons, alerts, recording, tim):
    """Draws status overlay on frame before streaming."""
    h, w = frame.shape[:2]

    colors = {
        STATE_IDLE:      (0, 200, 0),
        STATE_DETECTING: (0, 165, 255),
        STATE_ALERT:     (0, 0, 255),
        STATE_COOLDOWN:  (180, 0, 180),
    }
    sc = colors.get(state, (255, 255, 255))

    cv2.rectangle(frame, (0, 0), (w, 65), (0, 0, 0), -1)

    cv2.putText(frame, f"STATE: {state}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc, 2)
    cv2.putText(frame, f"Persons:{persons}  Alerts:{alerts}",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200,200,200), 1)
    cv2.putText(frame, datetime.now().strftime("%H:%M:%S"),
                (w-85, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130,130,130), 1)

    if state == STATE_DETECTING and tim > 0:
        p = min(tim / INTRUSION_TRIGGER_SECONDS, 1.0)
        cv2.rectangle(frame, (10,55), (w-10,63), (40,40,40), -1)
        cv2.rectangle(frame, (10,55), (10+int((w-20)*p),63), (0,165,255), -1)

    if recording:
        cv2.circle(frame, (w-22, 48), 9, (0,0,255), -1)
        cv2.putText(frame, "REC",
                    (w-70, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    return frame


# ── Public Controls ───────────────────────────────────────────────

def start_surveillance():
    """
    Starts the surveillance background thread.
    Call this once when app.py starts.
    """
    global _surveillance_thread

    if is_thread_running():
        print("[THREAD MGR] Thread already running.")
        return

    _stop_event.clear()   # reset stop signal

    _surveillance_thread = threading.Thread(
        target = _surveillance_loop,
        daemon = True,         # dies when main program exits
        name   = "SurveillanceThread"
    )
    _surveillance_thread.start()
    print("[THREAD MGR] Surveillance thread started.")


def stop_surveillance():
    """
    Signals the surveillance thread to stop cleanly.
    Call this on shutdown.
    """
    print("[THREAD MGR] Sending stop signal...")
    _stop_event.set()

    if _surveillance_thread:
        _surveillance_thread.join(timeout=5)   # wait max 5 seconds
        print("[THREAD MGR] Thread joined.")