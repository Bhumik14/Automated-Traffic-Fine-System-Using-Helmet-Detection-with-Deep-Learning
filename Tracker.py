from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self):
        self.object_tracker = DeepSort(
            max_age=20,
        )
    def track(self, detections, frame):
        tracking_ids = []
        boxes = []
        class_labels = [] 
        tracks = self.object_tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            tracking_ids.append(track.track_id)
            ltrb = track.to_ltrb()
            boxes.append(ltrb)
            class_labels.append(track.det_class)
            
        return tracking_ids, boxes, class_labels