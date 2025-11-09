
import cv2

def draw_boxes(img, detections, color=(0,255,0), thickness=2):
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        label = f"{d['label']}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, label, (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img

def put_hud(img, status_dict):
    y = 24
    for k, v in status_dict.items():
        txt = f"{k}: {'YES' if v else 'NO'}"
        cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 255), 2, cv2.LINE_AA)
        y += 28
    cv2.putText(img, 'q: quit  s: save frame', (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return img
