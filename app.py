
import argparse
import os
import time
from datetime import datetime
import cv2
import numpy as np
import pandas as pd

from detectors.yolo_detector import YoloDetector
from detectors.face_detector import FaceDetector
from detectors.smartwatch_heuristic import WristDeviceHeuristic
from utils.draw import draw_boxes, put_hud
from utils.events import EventLogger

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default="", help="Path to a video file. If empty, uses webcam (0).")
    ap.add_argument("--confidence", type=float, default=0.35, help="YOLO confidence threshold")
    ap.add_argument("--save-frames", action="store_true", help="Press 's' to save current frame to runs/frames")
    return ap.parse_args()

def main():
    args = parse_args()

    os.makedirs("runs", exist_ok=True)
    os.makedirs("runs/frames", exist_ok=True)

    # Init detectors
    yolo = YoloDetector(conf=args.confidence, classes_filter=["cell phone", "laptop", "book", "remote", "person"])
    face_det = FaceDetector()
    wrist_heur = WristDeviceHeuristic()

    # Video source
    cap = cv2.VideoCapture(0 if args.video == "" else args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera/video source")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    logger = EventLogger(csv_path="runs/events.csv")
    last_log_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        yolo_dets = yolo.detect(frame)
        face_dets = face_det.detect(frame)
        wrist_flags = wrist_heur.detect(frame)

        # Flags
        has_phone = any(d["label"] == "cell phone" for d in yolo_dets)
        has_face = len(face_dets) > 0
        wrist_device_suspect = wrist_flags.get("wrist_device", False)

        # Draw
        vis = frame.copy()
        vis = draw_boxes(vis, yolo_dets, color=(0, 255, 0))
        vis = draw_boxes(vis, face_dets, color=(255, 200, 0))
        status = {
            "phone": has_phone,
            "face": has_face,
            "wrist_device?": wrist_device_suspect
        }
        vis = put_hud(vis, status)

        # Log events every 1.5s if any flag is True
        now = time.time()
        if (has_phone or wrist_device_suspect or (not has_face)) and (now - last_log_time > 1.5):
            logger.log_event({
                "time": datetime.now().isoformat(timespec="seconds"),
                "phone": int(has_phone),
                "face_present": int(has_face),
                "wrist_device_suspect": int(wrist_device_suspect)
            })
            last_log_time = now

        cv2.imshow("Video Proctoring (q=quit, s=save frame)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') and args.save_frames:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = f"runs/frames/frame_{ts}.jpg"
            cv2.imwrite(out_path, vis)
            print(f"Saved {out_path}")

    cap.release()
    cv2.destroyAllWindows()
    logger.close()
    print("Done. Events saved to runs/events.csv")

if __name__ == "__main__":
    main()
