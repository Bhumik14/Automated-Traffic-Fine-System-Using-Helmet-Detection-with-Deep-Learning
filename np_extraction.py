import easyocr
import json
import numpy as np
import cv2

reader = easyocr.Reader(['en'], gpu=False)  # Use CPU to avoid GPU errors

def crop_image(frame, pbox):
    # Validate box coordinates before cropping
    x1, y1, x2, y2 = map(int, pbox)

    # Check bounds
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    # If crop is invalid (empty), return None
    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]

def ocr(img, bbox):
    cropImg = crop_image(img, bbox)
    if cropImg is None:
        print("[OCR] Skipping empty or invalid crop.")
        return ""
    
    img = cv2.imread(cropImg) if isinstance(cropImg, str) else cropImg
    upscaled_img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    if upscaled_img is None or upscaled_img.size == 0:
        print("[OCR] Skipping empty or invalid upscaled image.")
        return ""

    try:
        result = reader.readtext(upscaled_img)
        plate_text = result[0][1] if result else ""
        return plate_text
    except Exception as e:
        print(f"[OCR] Failed to read text: {e}")
        return ""
