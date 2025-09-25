import cv2
import numpy as np 
import torch
from YoloModel import YoloModel
from Tracker import Tracker
import os
from iou import iou
import json
from google import genai
from PIL import Image
from notify import send_notification

MODEL_PATH = './models/best.pt'
VIDEO_PATH = './assets/test2.mp4'

# Initialize Gemini client once
client = genai.Client(api_key="AIzaSyBuGk3kE8zdnFyhFwn8h8iEWF8YTXFUapA")

def extract_plate_text_with_gemini(image_path: str):
    """Send one number plate image to Gemini and return extracted text."""
    try:
        img = Image.open(image_path)
        response = client.models.generate_content(
            model="gemini-1.5-flash",   # use a multimodal model
            contents=[
                img,
                "Extract the license plate number from this image. Return strictly the plate text. Do not include spaces, symbols, or additional text."
            ],
        )
        return response.text.strip()
    except Exception as e:
        print("Gemini OCR failed:", e)
        return ""

def main():
    model = YoloModel(model_path=MODEL_PATH)
    tracker = Tracker()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = cap.get(cv2.CAP_PROP_FPS)

    output_video_path = 'output_tracked.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_fps, (frame_width, frame_height))

    os.makedirs("violations/riders", exist_ok=True)
    os.makedirs("violations/plates", exist_ok=True)

    best_plate = {"area": 0, "path": None}  # store best crop for Gemini

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        detections = model.detect(frame) or []
        tracking_ids, boxes, cls_labels = tracker.track(detections, frame)

        track_info = {}
        rider_boxes = {}

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
            cv2.putText(frame, f"{tracking_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Associate helmet/no_helmet/number_plate
        for obj_tracking_id, obj_box, obj_label in zip(tracking_ids, boxes, cls_labels):
            obj_box = list(map(int, obj_box))
            if obj_label not in ["helmet", "no_helmet", "number_plate"]:
                continue

            best_iou = 0
            best_rider_id = None
            for rider_id, rider_box in rider_boxes.items():
                overlap = iou(obj_box, rider_box)
                if overlap > best_iou and overlap > 0.01:
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
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        for tid, info in track_info.items():
            if info["rider"] and info["no_helmet"]:
                np_box = info["number_plate_bbox"]
                plate_path = None
                if np_box is not None:
                    x1, y1, x2, y2 = map(int, np_box)
                    plate_crop = frame[y1:y2, x1:x2]
                    plate_path = f"violations/plates/plate_{tid}_frame_{frame_count}.png"
                    cv2.imwrite(plate_path, plate_crop)

                    # track best plate (largest area = clearest)
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_plate["area"]:
                        best_plate = {"area": area, "path": plate_path}

                rider_crop = frame[info["rider_bbox"][1]:info["rider_bbox"][3],
                                   info["rider_bbox"][0]:info["rider_bbox"][2]]
                rider_path = f"violations/riders/rider_{tid}_frame_{frame_count}.png"
                cv2.imwrite(rider_path, rider_crop)

                violation_record = {
                    "track_id": tid,
                    "frame_time": cap.get(cv2.CAP_PROP_POS_MSEC),
                    "rider_img_path": rider_path,
                    "plate_img_path": plate_path
                }
                violations.append(violation_record)
                print("Violation recorded:", violation_record)

        with open("output_log.txt", "a") as f:
            for v in violations:
                f.write(json.dumps(v) + "\n")

        out.write(frame)

    cap.release()
    out.release()
    print(f"Tracking video saved to: {output_video_path}")

    # --- After video processing, send only the best plate to Gemini ---
    if best_plate["path"]:
        print("Sending best plate to Gemini:", best_plate["path"])
        plate_text = extract_plate_text_with_gemini(best_plate["path"])
        print("Extracted Plate Text:", plate_text)

        send_notification(plate_text);
    else:
        print("No plate detected to send to Gemini.")

if __name__ == "__main__":
    main()
