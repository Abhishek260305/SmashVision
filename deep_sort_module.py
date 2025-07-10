import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Initialize YOLO and DeepSORT
yolo = YOLO(r"X:\Users\Thunder\runs\yolov8l_badminton_1280\weights\best.pt")
tracker = DeepSort(max_age=30)

def run_deepsort(frame):
    """
    Runs YOLO + DeepSORT on the input frame.

    Returns:
        List of dicts with keys: 'track_id', 'bbox', 'centroid', 'cls'
    """
    results = yolo(frame)
    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id != 0:
            continue  # Only track players

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append(([x1, y1, x2-x1, y2-y1], conf, cls_id))

    tracks = tracker.update_tracks(detections, frame=frame)

    tracked_players = []
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = track.to_ltrb()
        cx = int(l + w / 2)
        cy = int(t + h / 2)

        tracked_players.append({
            'track_id': track_id,
            'bbox': (int(l), int(t), int(l+w), int(t+h)),
            'centroid': (cx, cy),
            'cls': 0
        })

    return tracked_players
