
# Video Proctoring ML (VS Code Ready)

A lightweight, local proctoring demo that detects **phone**, **face**, and **suspected wrist device (smartwatch?)** from webcam/video.
- Phone & common objects: YOLOv8n (Ultralytics)
- Face: MediaPipe Face Detection
- Wrist device (heuristic): MediaPipe Hands + simple edge-based heuristic around wrist landmarks

>  **Disclaimer**: The wrist-device detection is heuristic and not a robust "smartwatch classifier". Use it as a starting point.


 Outputs
- Live annotated window (`q` to quit)
- `runs/events.csv` with timestamps & flags (phone / face / wrist_device)
- `runs/frames/` (optional frame dumps when you press `s`)

## VS Code
- Open this folder
- Press **Ctrl+Shift+P → Python: Select Interpreter** and pick `.venv`
- Hit **F5** to debug with the provided launch config

## Known Labels (YOLOv8n)
- Uses COCO classes; includes `"cell phone"` for phones
- No built-in `"smartwatch"` class; we use a heuristic around wrist landmarks

## Notes
- For better accuracy, consider fine-tuning a YOLO model on private proctoring data (phones, watches, earbuds).
- You can swap YOLOv8n with `"yolov8s.pt"` for better accuracy (slower).
