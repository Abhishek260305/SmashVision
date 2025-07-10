import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_polygon(json_path):
    with open(json_path) as f:
        data = json.load(f)

    pts = []
    for d in data:
        if d["x"] is not None and d["y"] is not None:
            pts.append((d["x"], d["y"]))

    if len(pts) >= 3:
        pts_np = np.array(pts, dtype=np.float32)
        centroid = np.mean(pts_np, axis=0)
        angles = np.arctan2(pts_np[:,1] - centroid[1], pts_np[:,0] - centroid[0])
        sorted_pts = pts_np[np.argsort(angles)]
        closed_polygon = np.vstack([sorted_pts, sorted_pts[0]]).astype(np.int32)
        return closed_polygon
    else:
        print("[⚠️] Not enough points to form polygon.")
        return None

def load_keypoints(json_path):
    with open(json_path) as f:
        data = json.load(f)

    pts = []
    for d in data:
        if d["x"] is not None and d["y"] is not None:
            pts.append((d["x"], d["y"]))
    return np.array(pts, dtype=np.int32)

def point_inside_polygon(pt, polygon):
    if polygon is None:
        return False
    return cv2.pointPolygonTest(polygon, pt, False) >= 0

def load_mid_kps(json_path, mode="singles"):
    with open(json_path) as f:
        data = json.load(f)

    pts = []
    for d in data:
        if d["x"] is not None and d["y"] is not None:
            pts.append((d["x"], d["y"]))

    print(f"[DEBUG] Loaded {len(pts)} points from JSON: {pts}")

    if len(pts) >= 6:
        mid_kp1 = pts[4]
        mid_kp2 = pts[5]
    else:
        print("[❌] Not enough keypoints for midpoints.")
        return (None, None)

    return mid_kp1, mid_kp2

def plot_half_court_polygons(keypoints, mode="singles"):
    plt.figure(figsize=(8,6))
    plt.scatter(keypoints[:,0], keypoints[:,1], c='red')

    if mode == "singles":
        player1_poly = keypoints[[0,1,5,4,0]]
        plt.plot(player1_poly[:,0], player1_poly[:,1], 'b-', label='Player 1 court')

        player2_poly = keypoints[[4,5,3,2,4]]
        plt.plot(player2_poly[:,0], player2_poly[:,1], 'y-', label='Player 2 court')

    elif mode == "doubles":
        bottom_half = keypoints[[0,1,5,4,0]]
        top_half = keypoints[[4,5,3,2,4]]

        if bottom_half[:,1].mean() > top_half[:,1].mean():
            team1_poly = bottom_half
            team2_poly = top_half
        else:
            team1_poly = top_half
            team2_poly = bottom_half

        plt.plot(team1_poly[:,0], team1_poly[:,1], 'b-', label='Team 1 court (near)')
        plt.plot(team2_poly[:,0], team2_poly[:,1], 'y-', label='Team 2 court (far)')

    for i, pt in enumerate(keypoints):
        plt.text(pt[0], pt[1], str(i), fontsize=12)

    plt.legend()
    plt.title(f"Court Keypoints and {'Teams' if mode=='doubles' else 'Players'} Half-Court Polygons")
    plt.gca().invert_yaxis()
    plt.show()
