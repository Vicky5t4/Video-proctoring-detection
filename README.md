
# Video Proctoring ML (VS Code Ready)
<img width="1919" height="1079" alt="Screenshot 2025-11-09 112306" src="https://github.com/user-attachments/assets/295ded6f-abb1-4f85-990a-87cf4e7ccc1e" />
<img width="1914" height="1079" alt="Screenshot 2025-11-09 112348" src="https://github.com/user-attachments/assets/37b978d7-f8da-4f92-a4c6-2ea95171aa09" />



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
