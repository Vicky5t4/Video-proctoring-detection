
import pandas as pd
import os

class EventLogger:
    def __init__(self, csv_path='runs/events.csv'):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(csv_path):
            pd.DataFrame(columns=['time', 'phone', 'face_present', 'wrist_device_suspect']).to_csv(csv_path, index=False)
        self.buffer = []

    def log_event(self, event_dict):
        self.buffer.append(event_dict)
        if len(self.buffer) >= 10:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer)
        df.to_csv(self.csv_path, mode='a', header=False, index=False)
        self.buffer = []

    def close(self):
        self.flush()
