import cv2
import numpy as np 
import torch
from YoloModel import YoloModel
from Tracker import Tracker
import os
from iou import iou

MODEL_PATH = './models/best.pt'
VIDEO_PATH = './assets/test4.mp4'

def main():
    # Load YOLO model and tracker
    model = YoloModel(model_path=MODEL_PATH)
    tracker = Tracker()

    # Open input video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = cap.get(cv2.CAP_PROP_FPS)

    # Prepare output video writer
    output_video_path = 'output_tracked.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_fps, (frame_width, frame_height))

    # Process each frame
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        detections = model.detect(frame)
        detections = detections if detections is not None else []


        tracking_ids, boxes, cls_labels = tracker.track(detections, frame)

        # --- After you obtain tracking_ids, boxes, cls_labels ---
        track_info = {}
        rider_boxes = {}

        # First, process riders and build the mapping
        for tracking_id, bounding_box, cls_label in zip(tracking_ids, boxes, cls_labels):
            bounding_box = list(map(int, bounding_box))
            if cls_label == "rider":
                track_info[tracking_id] = {
                    "rider": True,
                    "helmet": False,
                    "no_helmet": False,
                    "number_plate": False,
                    "helmet_bbox": None,
                    "no_helmet_bbox": None,
                    "number_plate_bbox": None,
                    "rider_bbox": bounding_box,
                }
                rider_boxes[tracking_id] = bounding_box
        
            x1, y1, x2, y2 = map(int, bounding_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{tracking_id}", (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



        # Associate helmets, no_helmet, and number_plate to the closest rider by IoU
        for obj_tracking_id, obj_box, obj_label in zip(tracking_ids, boxes, cls_labels):
            obj_box = list(map(int, obj_box))

        
            if obj_label not in ["helmet", "no_helmet", "number_plate"]:
                continue

            # Find best rider match by IoU
            best_iou = 0
            best_rider_id = None
            for rider_id, rider_box in rider_boxes.items():
                overlap = iou(obj_box, rider_box)
                if overlap > best_iou and overlap > 0.01:  # Lower threshold for loose associations
                    best_iou = overlap
                    best_rider_id = rider_id

            if best_rider_id is not None:
                if obj_label == "helmet":
                    track_info[best_rider_id]["helmet"] = True
                    track_info[best_rider_id]["helmet_bbox"] = obj_box
                elif obj_label == "no_helmet":
                    track_info[best_rider_id]["no_helmet"] = True
                    track_info[best_rider_id]["no_helmet_bbox"] = obj_box
                elif obj_label == "number_plate":
                    track_info[best_rider_id]["number_plate"] = True
                    track_info[best_rider_id]["number_plate_bbox"] = obj_box

            


        violations = []
        for tid, info in track_info.items():
            if info["rider"] and info["no_helmet"]:
                violations.append({
                    "track_id": tid,
                    "rider_bbox": info["rider_bbox"],
                    "number_plate_bbox": info["number_plate_bbox"],
                    "frame_time": cap.get(cv2.CAP_PROP_POS_MSEC)
                })
        
        with open("output_log.txt", "a") as f:
            f.write(f"Voilation: {violations}")

        # os.makedirs("violations/riders", exist_ok=True)
        # os.makedirs("violations/plates", exist_ok=True)

        # frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # for v in violations:
        #     rbox = list(map(int, v["rider_bbox"]))  # convert floats to ints
        #     rider_crop = frame[rbox[1]:rbox[3], rbox[0]:rbox[2]]

        #     rider_path = f"violations/riders/rider_{v['track_id']}_frame_{frame_count}.png"
        #     cv2.imwrite(rider_path, rider_crop)

        #     if v["number_plate_bbox"] is not None:
        #         pbox = list(map(int, v["number_plate_bbox"]))
        #         plate_crop = frame[pbox[1]:pbox[3], pbox[0]:pbox[2]]
        #         plate_path = f"violations/plates/plate_{v['track_id']}_frame_{frame_count}.png"
        #         cv2.imwrite(plate_path, plate_crop)
        #     else:
    
        #         plate_path = None

        #     # You can save metadata to a JSON or database as needed here
        #     violation_record = {
        #         "track_id": v["track_id"],
        #         "frame_time": v["frame_time"],
        #         "rider_img_path": rider_path,
        #         "plate_img_path": plate_path
        #     }
        #     print("Violation recorded:", violation_record)

        # Optionally save violation_record to persistent storage
        print("Violations", violations)
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Tracking video saved to: {output_video_path}")

if __name__ == "__main__":
    main()
