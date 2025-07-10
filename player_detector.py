# player_detector.py

from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, yolo_model_path):
        self.model = YOLO(yolo_model_path)

    def detect(self, frame):
        """
        Detect persons and shuttle in frame.
        Returns: list of person bboxes, list of shuttle bboxes
        """
        results = self.model(frame)

        persons = []
        shuttles = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                if cls_id == 0:  # person
                    persons.append((x1,y1,x2,y2))
                elif cls_id == 1:  # shuttle
                    shuttles.append((x1,y1,x2,y2))

        return persons, shuttles
