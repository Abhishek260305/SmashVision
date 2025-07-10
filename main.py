import cv2
import time
import threading
import queue
import numpy as np
from collections import deque
from ultralytics import YOLO
from court_module import CourtNetModule
from shuttle_utils import load_polygon, point_inside_polygon, load_mid_kps, load_keypoints
from deep_sort_realtime.deepsort_tracker import DeepSort
from kalman6D import KalmanFilter
from net_utils import load_net_keypoints, net_touch_event

# === Threaded video reader ===
class VideoCaptureThreaded:
    def __init__(self, path, queue_size=10):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"[ERROR] Failed to open video: {path}")
        self.q = queue.Queue(maxsize=queue_size)
        self.running = True
        self.thread = threading.Thread(target=self.reader, daemon=True)
        self.thread.start()

    def reader(self):
        while self.running:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.running = False
                    break
                self.q.put(frame)
            else:
                time.sleep(0.005)

    def read(self):
        if not self.q.empty():
            return True, self.q.get()
        elif not self.running:
            return False, None
        else:
            time.sleep(0.005)
            return True, None

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# === Utility: Assign point by nearest keypoint ===
def assign_point_by_nearest_kp(shuttle_xy, keypoints, mode="singles"):
    if mode == "singles" and len(keypoints) >= 6:
        p1_kps = [keypoints[i] for i in [0,1,5,4,0]]
        p2_kps = [keypoints[i] for i in [4,5,3,2,4]]

        dist_p1 = np.mean([np.linalg.norm(np.array(shuttle_xy) - np.array(kp)) for kp in p1_kps])
        dist_p2 = np.mean([np.linalg.norm(np.array(shuttle_xy) - np.array(kp)) for kp in p2_kps])

        return "Player 1" if dist_p1 < dist_p2 else "Player 2"
    return None

# === Utility: Pixel-to-meter ratio based on court width ===
def compute_pixel_to_meter_ratio(keypoints):
    # Badminton court width is 6.1 meters (singles and doubles same baseline width)
    court_width_m = 6.1
    if len(keypoints) >= 6:
        # Using baseline side keypoints indices for singles: [0] left, [2] right
        left_kp = keypoints[0]
        right_kp = keypoints[2]
        pixel_width = np.linalg.norm(np.array(left_kp) - np.array(right_kp))
        if pixel_width > 0:
            return court_width_m / pixel_width
    return None

# === Config ===
video_path = r"X:\Users\Thunder\Downloads (X)\testCase1.mp4"
court_model_path = r"X:\Projects\thread\out_COURT_kprcnn_combined\model_final.pth"
net_model_path = r"X:\Projects\thread\out_NET_kprcnn_combined\model_final.pth"
yolo_model_path = r"X:\Users\Thunder\runs\yolov8l_badminton_1280\weights\best.pt"
polygon_path = r"X:\Projects\thread\Research\output\court_polygon.json"
net_polygon_path = r"X:\Projects\thread\Research\output\net_polygon.json"

output_res = (1280, 720)
output_video_path = r"X:\Projects\thread\Research\video_outputs\testCase1_output.mp4"

# === Initialize models ===
court_net = CourtNetModule(court_model_path, net_model_path)
yolo = YOLO(yolo_model_path)
tracker = DeepSort(max_age=30)
kf = KalmanFilter(fps=30)
net_keypoints = load_net_keypoints(net_polygon_path)

# === Initialize video reader ===
cap = VideoCaptureThreaded(video_path, queue_size=20)
fps = 30
frame_idx = 0
mode = None

# === Initialize VideoWriter ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, output_res)

# === Rally variables ===
rally_active = False
rally_start_time = None
last_status = None
no_detect_count = 0

# === Shuttle rest detection ===
shuttle_positions = deque(maxlen=5)
rest_threshold = 5

# === Scores ===
player1_score = 0
player2_score = 0

# === Point cooldown ===
point_cooldown_frames = 0
point_cooldown_threshold = 30  # Adjust as needed (e.g. 30 frames = ~1 second at 30fps)

# === Speed tracking ===
prev_shuttle_pos = None
pixel_to_meter_ratio = None
# === Player speed tracking ===
prev_player_positions = {}  # {track_id: (cx, cy)}

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of video.")
        break
    if frame is None:
        continue

    frame_resized = cv2.resize(frame, output_res)

    # === YOLO detection ===
    results = yolo(frame_resized, verbose=False)
    persons = [b for b in results[0].boxes if int(b.cls[0]) == 0]
    shuttles = [b for b in results[0].boxes if int(b.cls[0]) == 1]

    # === Determine mode ===
    if mode is None:
        mode = "singles" if len(persons) <= 2 else "doubles"
        print(f"[INFO] Mode: {mode}")

    # === Court keypoints ===
    if frame_idx == 0:
        court_net.detect_keypoints(frame_resized, mode=mode)
        court_net.save_keypoints_json(output_res, output_res, path=polygon_path)

    frame_resized = court_net.draw_keypoints(frame_resized, mode=mode)
    keypoints = load_keypoints(polygon_path)

    # === Compute pixel-to-meter ratio once ===
    if pixel_to_meter_ratio is None:
        pixel_to_meter_ratio = compute_pixel_to_meter_ratio(keypoints)
        print(f"[INFO] Pixel-to-meter ratio: {pixel_to_meter_ratio:.5f} m/pixel" if pixel_to_meter_ratio else "[WARN] Could not compute pixel-to-meter ratio.")

    mid_kp1, mid_kp2 = load_mid_kps(polygon_path, mode=mode)
    if mid_kp1 and mid_kp2:
        cv2.line(frame_resized,
                 (int(mid_kp1[0]), int(mid_kp1[1])),
                 (int(mid_kp2[0]), int(mid_kp2[1])),
                 (255, 0, 255), 2)

    # === Deep SORT tracking ===
    dets_for_tracker = []
    for person in persons:
        x1, y1, x2, y2 = map(int, person.xyxy[0])
        conf = float(person.conf[0])
        dets_for_tracker.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    tracks = tracker.update_tracks(dets_for_tracker, frame=frame_resized)

    players = []
    for track in tracks:
        if track.is_confirmed():
            tid, l, t, r, b = track.track_id, *track.to_ltrb()
            players.append((tid, l, t, r, b, int(b)))
    players_sorted = sorted(players, key=lambda x: x[5], reverse=True)

    if mode == "singles" and len(players_sorted) >= 2:
        for idx, p, color, label in [(1, players_sorted[0], (255, 0, 0), "Player1"),
                                    (2, players_sorted[1], (0, 255, 255), "Player2")]:
            track_id, l, t, r, b, _ = p
            bottom_cx, bottom_cy = (l + r) // 2, b

        # Default arrow parameters
            arrow_length = (r - l) / 3  # fixed 1/3 of BB width
            arrow_length = max(arrow_length, 5)  # ensure minimum length for visibility

            if track_id in prev_player_positions and pixel_to_meter_ratio:
                prev_bottom_cx, prev_bottom_cy = prev_player_positions[track_id]
                dx = bottom_cx - prev_bottom_cx
                dy = bottom_cy - prev_bottom_cy
                pixel_dist = np.sqrt(dx**2 + dy**2)
                meter_dist = pixel_dist * pixel_to_meter_ratio
                speed_mps = meter_dist * fps

            # Determine direction for arrow head
                if dx > 0:
                # moving right
                    arrow_start = (int(bottom_cx - arrow_length/2), int(bottom_cy + 5))
                    arrow_end = (int(bottom_cx + arrow_length/2), int(bottom_cy + 5))
                else:
                # moving left
                    arrow_start = (int(bottom_cx + arrow_length/2), int(bottom_cy + 5))
                    arrow_end = (int(bottom_cx - arrow_length/2), int(bottom_cy + 5))

                cv2.arrowedLine(frame_resized, arrow_start, arrow_end, color, 2, tipLength=0.3)
            else:
                speed_mps = 0.0

            prev_player_positions[track_id] = (bottom_cx, bottom_cy)

        # Draw BB and speed text
            cv2.rectangle(frame_resized, (int(l), int(t)), (int(r), int(b)), color, 2)
            cv2.putText(frame_resized, f"{label}: {speed_mps:.2f} m/s", (int(l), int(t)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # === Shuttle detection ===
    if shuttles:
        no_detect_count = 0
        if not rally_active:
            rally_active = True
            rally_start_time = time.time()
            last_status = None
            shuttle_positions.clear()

        shuttle = shuttles[0]
        x1, y1, x2, y2 = map(int, shuttle.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        kf.step([cx, cy])
        cx_filt, cy_filt = kf.S[0], kf.S[3]

        # === Shuttle speed computation ===
        if prev_shuttle_pos is not None and pixel_to_meter_ratio:
            dx = cx_filt - prev_shuttle_pos[0]
            dy = cy_filt - prev_shuttle_pos[1]
            pixel_dist = np.sqrt(dx**2 + dy**2)
            meter_dist = pixel_dist * pixel_to_meter_ratio
            shuttle_speed = meter_dist * fps  # m/s
        else:
            shuttle_speed = 0.0

        prev_shuttle_pos = (cx_filt, cy_filt)

        # === Net touch detection ===
        if net_touch_event((cx_filt, cy_filt), net_keypoints, y_thresh=10):
            cv2.putText(frame_resized, "Net Touch Detected!", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # === Half court polygons ===
        if mode == "singles" and len(keypoints) >= 6:
            p1_poly = np.array([keypoints[i] for i in [0, 1, 5, 4]], dtype=np.int32)
            p2_poly = np.array([keypoints[i] for i in [4, 5, 3, 2]], dtype=np.int32)
        else:
            p1_poly = p2_poly = None

        # === Shuttle side ===
        if point_inside_polygon((cx_filt, cy_filt), p1_poly):
            shuttle_color = (255, 0, 0)
            side_status = "Player 1 side"
        elif point_inside_polygon((cx_filt, cy_filt), p2_poly):
            shuttle_color = (0, 255, 255)
            side_status = "Player 2 side"
        else:
            shuttle_color = (0, 0, 255)
            side_status = "Outside"

        # === Draw shuttle BB, kalman dot, and speed ===
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), shuttle_color, 2)
        cv2.circle(frame_resized, (int(cx_filt), int(cy_filt)), 5, shuttle_color, -1)
        cv2.putText(frame_resized, side_status, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, shuttle_color, 2)
        cv2.putText(frame_resized, f"{shuttle_speed:.2f} m/s", (x1, y1-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, shuttle_color, 2)

        # === Shuttle rest detection with cooldown ===
        shuttle_positions.append((cx_filt, cy_filt))
        if len(shuttle_positions) == shuttle_positions.maxlen:
            max_dist = max(
                np.linalg.norm(np.array(p1) - np.array(p2))
                for i, p1 in enumerate(shuttle_positions)
                for p2 in list(shuttle_positions)[i+1:]
            )
            if max_dist < rest_threshold:
                if point_cooldown_frames == 0:
                    if side_status == "Player 1 side":
                        player2_score += 1
                    elif side_status == "Player 2 side":
                        player1_score += 1
                    else:
                        winner = assign_point_by_nearest_kp((cx_filt, cy_filt), keypoints, mode=mode)
                        if winner == "Player 1":
                            player1_score += 1
                        elif winner == "Player 2":
                            player2_score += 1

                    rally_active = False
                    last_status = side_status
                    shuttle_positions.clear()
                    point_cooldown_frames = point_cooldown_threshold  # Start cooldown

    else:
        kf.step([None, None])
        no_detect_count += 1
        if no_detect_count >= 7 and rally_active:
            rally_active = False
            last_status = "Shuttle Lost"
            shuttle_positions.clear()

    # === Cooldown update ===
    if point_cooldown_frames > 0:
        point_cooldown_frames -= 1

    # === Overlay scoreboard and rally status ===
    if rally_active and rally_start_time:
        cv2.putText(frame_resized, f"Rally Time: {time.time() - rally_start_time:.2f}s",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        cv2.putText(frame_resized, f"Rally Over ({last_status})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame_resized, f"P2: {player2_score}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.putText(frame_resized, f"P1: {player1_score}", (20, output_res[1] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    # === Debug: Cooldown counter ===
    cv2.putText(frame_resized, f"Cooldown: {point_cooldown_frames}",
                (output_res[0]-250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # === Write frame to output video ===
    out.write(frame_resized)

    cv2.imshow("Badminton Optimized Pipeline", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
