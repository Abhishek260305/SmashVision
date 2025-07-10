import numpy as np
import cv2

def assign_player_ids(deep_sort_tracks, polygon, mode="singles"):
    """
    Assigns player IDs (1 or 2) based on position in court half.

    Parameters:
    - deep_sort_tracks: list of dicts with keys ['track_id', 'bbox', 'centroid']
    - polygon: list of court keypoints
    - mode: "singles" or "doubles"

    Returns:
    - List of player dicts with added 'player_id'
    """

    players = []

    for track in deep_sort_tracks:
        cx, cy = track['centroid']

        # Determine which half player is in
        if mode == "singles":
            # player2 half: inside kps 6,7,10,11
            p2_poly = np.array([polygon[6], polygon[7], polygon[10], polygon[11]], dtype=np.int32)
            # player1 half: inside kps 4,5,10,11
            p1_poly = np.array([polygon[4], polygon[5], polygon[10], polygon[11]], dtype=np.int32)

            if cv2.pointPolygonTest(p2_poly, (cx, cy), False) >= 0:
                player_id = 2
            else:
                player_id = 1

        else:  # doubles
            # player2 half: inside kps 2,3,8,9
            p2_poly = np.array([polygon[2], polygon[3], polygon[8], polygon[9]], dtype=np.int32)
            # player1 half: inside kps 0,1,8,9
            p1_poly = np.array([polygon[0], polygon[1], polygon[8], polygon[9]], dtype=np.int32)

            if cv2.pointPolygonTest(p2_poly, (cx, cy), False) >= 0:
                player_id = 2
            else:
                player_id = 1

        track['player_id'] = player_id
        players.append(track)

    # Enforce only ids 1 and 2
    if mode == "singles" and len(players) > 2:
        players = sorted(players, key=lambda x: x['centroid'][1])  # sort by y (top to bottom)
        players = players[:2]  # keep only top 2 players

    elif mode == "doubles" and len(players) > 4:
        # Keep top 2 and bottom 2 as two teams
        players = sorted(players, key=lambda x: x['centroid'][1])
        players = players[:4]

    return players

def color_shuttle_by_half(shuttle_pos, polygon, mode="singles"):
    """
    Returns shuttle color (BGR) based on which half it's in.
    Blue for Player 1 half, Yellow for Player 2 half, Red for outside.
    """

    cx, cy = shuttle_pos

    if mode == "singles":
        p2_poly = np.array([polygon[6], polygon[7], polygon[10], polygon[11]], dtype=np.int32)
        p1_poly = np.array([polygon[4], polygon[5], polygon[10], polygon[11]], dtype=np.int32)
    else:
        p2_poly = np.array([polygon[2], polygon[3], polygon[8], polygon[9]], dtype=np.int32)
        p1_poly = np.array([polygon[0], polygon[1], polygon[8], polygon[9]], dtype=np.int32)

    if cv2.pointPolygonTest(p1_poly, (cx, cy), False) >= 0:
        return (255, 0, 0)  # Blue for player 1
    elif cv2.pointPolygonTest(p2_poly, (cx, cy), False) >= 0:
        return (0, 255, 255)  # Yellow for player 2
    else:
        return (0, 0, 255)  # Red if outside
