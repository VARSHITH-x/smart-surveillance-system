# config.py — UNIFIED for both Student 1 and Student 2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Camera ────────────────────────────────────────────────────────
CAMERA_INDEX  = 0
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
FPS           = 20
TARGET_FPS    = 20   # alias for friend's code

# ── YOLO ──────────────────────────────────────────────────────────
YOLO_MODEL            = os.path.join(BASE_DIR, "models", "yolov8n.pt")
CONFIDENCE_THRESHOLD  = 0.5
YOLO_CONFIDENCE       = 0.5    # alias for friend's code
PERSON_CLASS_ID       = 0
YOLO_CLASSES          = [0]    # alias for friend's code

# ── Motion ────────────────────────────────────────────────────────
MOTION_THRESHOLD          = 5000   # use YOUR value (500 is too sensitive)
INTRUSION_TRIGGER_SECONDS = 2
INTRUSION_SECONDS         = 2      # alias for friend's code

# ── Alerts ────────────────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS = 30
SCREENSHOT_PREFIX      = "alert"
CLIP_DURATION_SECONDS  = 10
CLIP_DURATION          = 10        # alias for friend's code

# ── Folders ───────────────────────────────────────────────────────
SCREENSHOT_FOLDER  = os.path.join(BASE_DIR, "static", "screenshots")
SCREENSHOTS_FOLDER = os.path.join(BASE_DIR, "static", "screenshots")  # alias
VIDEO_FOLDER       = os.path.join(BASE_DIR, "static", "videos")
VIDEOS_FOLDER      = os.path.join(BASE_DIR, "static", "videos")       # alias

# ── Database ──────────────────────────────────────────────────────
# Use logs/ folder (your structure)
DATABASE_PATH = os.path.join(BASE_DIR, "logs", "surveillance.db")

# ── Flask ─────────────────────────────────────────────────────────
FLASK_HOST         = "0.0.0.0"
FLASK_PORT         = 5000
DEBUG_MODE         = False
SECRET_KEY         = "surveillance_secret_key_2024"
MAX_ALERTS_DISPLAY = 10

# ── Class-based alias (for friend's code that uses Config.SOMETHING) ──
class Config:
    SECRET_KEY         = SECRET_KEY
    DATABASE_PATH      = DATABASE_PATH
    SCREENSHOTS_FOLDER = SCREENSHOTS_FOLDER
    VIDEOS_FOLDER      = VIDEOS_FOLDER
    MAX_ALERTS_DISPLAY = MAX_ALERTS_DISPLAY
    CAMERA_INDEX       = CAMERA_INDEX
    FRAME_WIDTH        = FRAME_WIDTH
    FRAME_HEIGHT       = FRAME_HEIGHT
    TARGET_FPS         = TARGET_FPS
    YOLO_MODEL         = YOLO_MODEL
    YOLO_CONFIDENCE    = YOLO_CONFIDENCE
    YOLO_CLASSES       = YOLO_CLASSES
    MOTION_THRESHOLD   = MOTION_THRESHOLD
    INTRUSION_SECONDS  = INTRUSION_SECONDS
    CLIP_DURATION      = CLIP_DURATION
    FLASK_HOST         = FLASK_HOST
    FLASK_PORT         = FLASK_PORT