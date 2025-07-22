import cv2
import numpy as np 
import torch
from YoloModel import YoloModel
from Tracker import Tracker

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

        tracking_ids, boxes = tracker.track(detections, frame)

        for tracking_id, bounding_box in zip(tracking_ids, boxes):
            x1, y1, x2, y2 = map(int, bounding_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{tracking_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshpw(frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Tracking video saved to: {output_video_path}")

if __name__ == "__main__":
    main()
