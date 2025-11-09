
import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(min_detection_confidence=min_detection_confidence)

    def detect(self, frame):
        h, w = frame.shape[:2]
        # Convert to RGB for mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.detector.process(rgb)
        dets = []
        if res.detections:
            for d in res.detections:
                bbox = d.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                dets.append({"label": "face", "conf": d.score[0] if d.score else 0.5, "bbox": [x1, y1, x2, y2]})
        return dets
