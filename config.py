# config.py

# --- Camera ---
CAMERA_INDEX   = 0
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
FPS            = 20

# --- YOLO ---
YOLO_MODEL             = "models/yolov8n.pt"
CONFIDENCE_THRESHOLD   = 0.5
PERSON_CLASS_ID        = 0

# --- Motion ---
MOTION_THRESHOLD           = 5000
INTRUSION_TRIGGER_SECONDS  = 2

# --- Alerts ---
ALERT_COOLDOWN_SECONDS   = 30
SCREENSHOT_FOLDER        = "static/screenshots"
SCREENSHOT_PREFIX        = "alert"
VIDEO_FOLDER             = "static/videos"
CLIP_DURATION_SECONDS    = 10

# --- Database ---
DATABASE_PATH = "logs/surveillance.db"

# --- Flask ---
FLASK_HOST  = "0.0.0.0"
FLASK_PORT  = 5000
DEBUG_MODE  = False