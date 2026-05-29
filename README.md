# AI-Powered Real-Time Smart Surveillance System

A software-only intelligent surveillance system built with Python, OpenCV, and YOLOv8.
Runs entirely on a laptop webcam — no external hardware required.

---

## Features

- Live webcam feed with real-time AI processing
- Human detection using YOLOv8 (pretrained model)
- Motion detection using frame differencing
- Intrusion alert system with state machine logic
- Automatic screenshot capture on alert
- Video clip recording during intrusion events
- SQLite database logging of all alert events
- Flask web dashboard with live MJPEG stream
- Multithreaded architecture for real-time performance

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Computer Vision | OpenCV |
| AI Detection | YOLOv8 (Ultralytics) |
| Backend | Flask |
| Database | SQLite |
| Concurrency | Python threading |
| Frontend | HTML, CSS |

---

## Project Architecture