# court_module.py

import cv2
import numpy as np
import json
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

class CourtNetModule:
    def __init__(self, court_model_path, net_model_path):
        self.court_predictor = self.load_model(court_model_path, 12)
        self.net_predictor = self.load_model(net_model_path, 5)
        self.last_court_kp = None
        self.last_net_kp = None

    def load_model(self, model_path, num_keypoints):
        cfg = get_cfg()
        cfg.merge_from_file("detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = num_keypoints
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        return DefaultPredictor(cfg)

    def detect_keypoints(self, frame, mode="singles"):
        # Detect court keypoints
        outputs = self.court_predictor(frame)
        instances = outputs["instances"].to("cpu")
        if instances.has("pred_keypoints"):
            kp = instances.pred_keypoints.numpy()[0]
            if mode == "singles":
                indices = [4,5,6,7,10,11]
            else:
                indices = [0,1,2,3,8,9]
            self.last_court_kp = [kp[i] for i in indices]

        # Detect net keypoints
        outputs = self.net_predictor(frame)
        instances = outputs["instances"].to("cpu")
        if instances.has("pred_keypoints"):
            self.last_net_kp = instances.pred_keypoints.numpy()[0]

    def draw_keypoints(self, frame, mode="singles"):
        # Draw court keypoints
        if self.last_court_kp is not None:
            pts = []
            for x, y, score in self.last_court_kp:
                if score > 0.3:
                    pts.append((int(x), int(y)))
                    cv2.circle(frame, (int(x), int(y)), 4, (0,255,0), -1)

            if len(pts) >= 3:
                pts_np = np.array(pts, dtype=np.float32)
                centroid = np.mean(pts_np, axis=0)
                angles = np.arctan2(pts_np[:,1]-centroid[1], pts_np[:,0]-centroid[0])
                sorted_pts = pts_np[np.argsort(angles)]
                closed_polygon = np.vstack([sorted_pts, sorted_pts[0]]).astype(np.int32).reshape((-1,1,2))
                cv2.polylines(frame, [closed_polygon], isClosed=True, color=(0,255,0), thickness=2)

        # Draw net keypoints
        if self.last_net_kp is not None:
            pts = []
            for x, y, score in self.last_net_kp:
                if score > 0.3:
                    pts.append((int(x), int(y)))
                    cv2.circle(frame, (int(x), int(y)), 3, (0,0,255), -1)

            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    cv2.line(frame, pts[i], pts[j], (0,0,255), 1)

        return frame

    def save_keypoints_json(self, input_res, target_res, path):
        """
        Save adjusted court keypoints.
        """
        if self.last_court_kp is None:
            print("[WARN] No court keypoints to save.")
            return

        scale_x = target_res[0] / input_res[0]
        scale_y = target_res[1] / input_res[1]

        adjusted = []
        for x, y, score in self.last_court_kp:
            adj_x = int(x * scale_x)
            adj_y = int(y * scale_y)
            adjusted.append({"x": adj_x, "y": adj_y, "score": float(score)})

        with open(path, "w") as f:
            json.dump(adjusted, f, indent=4)
        print(f"[✅] Saved adjusted court polygon to {path}")

    def save_net_keypoints_json(self, input_res, target_res, path):
        """
        Save adjusted net keypoints.
        """
        if self.last_net_kp is None:
            print("[WARN] No net keypoints to save.")
            return

        scale_x = target_res[0] / input_res[0]
        scale_y = target_res[1] / input_res[1]

        adjusted = []
        for x, y, score in self.last_net_kp:
            adj_x = int(x * scale_x)
            adj_y = int(y * scale_y)
            adjusted.append({"x": adj_x, "y": adj_y, "score": float(score)})

        with open(path, "w") as f:
            json.dump(adjusted, f, indent=4)
        print(f"[✅] Saved adjusted net polygon to {path}")
