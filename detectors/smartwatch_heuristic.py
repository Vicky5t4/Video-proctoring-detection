
import cv2
import numpy as np
import mediapipe as mp

class WristDeviceHeuristic:
    '''
    Heuristic: use MediaPipe Hands, take a small patch around wrist landmarks,
    compute edge density and rectangularity score. If above thresholds, flag 'wrist_device'.
    This is NOT a robust smartwatch classifier—just a simple signal.
    '''
    def __init__(self, edge_thresh=0.12):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(model_complexity=0, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.edge_thresh = edge_thresh

    def _edge_density(self, patch):
        g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3,3), 0)
        edges = cv2.Canny(g, 60, 120)
        return edges.mean() / 255.0  # 0..1

    def detect(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        wrist_device = False

        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                # wrist index 0
                wx = int(hl.landmark[0].x * w)
                wy = int(hl.landmark[0].y * h)
                # crop 80x80 around wrist (clamped to image)
                sz = 80
                x1 = max(0, wx - sz//2); y1 = max(0, wy - sz//2)
                x2 = min(w, wx + sz//2); y2 = min(h, wy + sz//2)
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue
                patch = frame[y1:y2, x1:x2]
                ed = self._edge_density(patch)
                if ed > self.edge_thresh:
                    wrist_device = True
                    # Optional: draw a rectangle for visualization (handled by higher-level draw if needed)
        return {"wrist_device": wrist_device}
