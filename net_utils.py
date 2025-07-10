import json
import numpy as np

def load_net_keypoints(json_path):
    """
    Load net keypoints as (5,2) numpy array sorted by angle from centroid
    """
    with open(json_path) as f:
        data = json.load(f)

    pts = []
    for d in data:
        if d["x"] is not None and d["y"] is not None:
            pts.append((d["x"], d["y"]))

    pts_np = np.array(pts, dtype=np.float32)
    if pts_np.shape[0] < 3:
        print("[⚠️] Not enough net keypoints for polygon.")
        return None

    # Centroid + angular sort for consistent order
    centroid = np.mean(pts_np, axis=0)
    angles = np.arctan2(pts_np[:,1] - centroid[1], pts_np[:,0] - centroid[0])
    sorted_pts = pts_np[np.argsort(angles)]

    return sorted_pts

def net_touch_event(shuttle_xy, net_kps, y_thresh=10):
    """
    Detects if shuttle touches net based on y-coordinate proximity to net keypoints.
    Args:
        shuttle_xy: (cx, cy) of shuttle
        net_kps: net keypoints (5,2)
        y_thresh: vertical pixel threshold for net touch
    Returns:
        True if touch detected, else False
    """
    if net_kps is None:
        return False

    # Compute mean net y (or min/max based on your use-case)
    net_y_mean = net_kps[:,1].mean()

    cx, cy = shuttle_xy
    if abs(cy - net_y_mean) <= y_thresh:
        return True
    return False
