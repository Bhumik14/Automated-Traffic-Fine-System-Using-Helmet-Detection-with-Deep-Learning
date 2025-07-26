import easyocr 
import json
reader = easyocr.Reader(['en'])

def crop_image(img, bbox):
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2]

def ocr(img, bbox):
    crop_img = crop_image(img, bbox)
    result = reader.readtext(img)
    plate_text = result[0][1] if result else ""
    return plate_text