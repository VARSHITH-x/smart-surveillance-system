# database.py
import sqlite3
import os
from datetime import datetime
from config import DATABASE_PATH


def get_connection():
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT    NOT NULL,
            person_count     INTEGER DEFAULT 0,
            screenshot_path  TEXT,
            video_path       TEXT,
            duration_seconds REAL    DEFAULT 0,
            notes            TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_logs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            level     TEXT NOT NULL,
            message   TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_stats (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            date          TEXT    UNIQUE NOT NULL,
            total_alerts  INTEGER DEFAULT 0,
            total_persons INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Database ready at: {DATABASE_PATH}")


def log_alert(timestamp, person_count, screenshot_path=None,
              video_path=None, duration_seconds=0, notes=None):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO alerts
            (timestamp, person_count, screenshot_path,
             video_path, duration_seconds, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (timestamp, person_count, screenshot_path,
          video_path, duration_seconds, notes))
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    print(f"[DB] Alert #{new_id} saved.")
    return new_id


def get_recent_alerts(limit=20):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM alerts ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_alert_count():
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as total FROM alerts")
    result = cursor.fetchone()
    conn.close()
    return result["total"] if result else 0


def get_todays_alerts():
    conn   = get_connection()
    cursor = conn.cursor()
    today  = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("""
        SELECT * FROM alerts WHERE timestamp LIKE ?
        ORDER BY id DESC
    """, (f"{today}%",))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def log_system_event(level, message):
    conn      = get_connection()
    cursor    = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO system_logs (timestamp, level, message)
        VALUES (?, ?, ?)
    """, (timestamp, level, message))
    conn.commit()
    conn.close()


def update_daily_stats(person_count):
    conn   = get_connection()
    cursor = conn.cursor()
    today  = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("""
        INSERT OR IGNORE INTO daily_stats (date, total_alerts, total_persons)
        VALUES (?, 0, 0)
    """, (today,))
    cursor.execute("""
        UPDATE daily_stats
        SET total_alerts  = total_alerts  + 1,
            total_persons = total_persons + ?
        WHERE date = ?
    """, (person_count, today))
    conn.commit()
    conn.close()


def get_dashboard_stats():
    return {
        "total_alerts":  get_alert_count(),
        "todays_alerts": len(get_todays_alerts()),
        "recent_alerts": get_recent_alerts(10),
    }


# ── TEST ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    initialize_database()
    id1 = log_alert("2024-01-15 14:30:00", 1, "static/screenshots/test.jpg")
    log_system_event("INFO", "Test event")
    update_daily_stats(1)
    print("\nRecent alerts:")
    for a in get_recent_alerts():
        print(f"  {dict(a)}")
    print(f"\nTotal: {get_alert_count()}")
    print(f"DB file exists: {os.path.exists(DATABASE_PATH)}")
    print("[TEST] database.py OK")