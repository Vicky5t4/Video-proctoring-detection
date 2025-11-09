
from ultralytics import YOLO
import numpy as np

class YoloDetector:
    def __init__(self, model_name='yolov8n.pt', conf=0.35, classes_filter=None):
        self.model = YOLO(model_name)  # downloads if missing
        self.conf = conf
        self.classes_filter = set(classes_filter) if classes_filter else None

    def detect(self, frame):
        # Ultralytics expects RGB
        results = self.model.predict(source=frame[:, :, ::-1], imgsz=640, conf=self.conf, verbose=False)
        dets = []
        for r in results:
            boxes = r.boxes
            names = r.names
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                label = names.get(cls_id, str(cls_id))
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                if self.classes_filter and label not in self.classes_filter:
                    continue
                dets.append({"label": label, "conf": conf, "bbox": [x1, y1, x2, y2]})
        return dets
