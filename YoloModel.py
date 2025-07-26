from ultralytics import YOLO

class YoloModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.classList = ['helmet', 'no_helmet', 'number_plate', 'rider']

    def detect(self, image):
        results = self.model(image)
        detections = self.make_detections(results[0])
        return detections

    def make_detections(self, result):
        detections = []

        for box in result.boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            class_num = int(box.cls[0])


            # Optional: filter unwanted classes
            if result.names[class_num] not in self.classList:
                continue
            conf = box.conf[0]
            # Append in correct format
            cls_label = result.names[class_num]
            detections.append((([x1, y1, w, h]), conf, cls_label ))

        return detections
    
