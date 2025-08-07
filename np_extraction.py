import easyocr 
import json
reader = easyocr.Reader(['en'])

def crop_image(frame, pbox):
    plate_crop = frame[pbox[1]:pbox[3], pbox[0]:pbox[2]]
    
    return plate_crop

def ocr(img, bbox):
    crop_img = crop_image(img, bbox)
    result = reader.readtext(img)
    plate_text = result[0][1] if result else ""
    return plate_text