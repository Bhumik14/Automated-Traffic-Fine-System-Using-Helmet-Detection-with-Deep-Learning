from google import genai
from PIL import Image
import io

def extract_number_plate(image_path: str, prompt: str = "Extract the license plate text from this image."):
    client = genai.Client()

    img = Image.open(image_path)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[img, prompt]
    )

    return response.text

    # return "ABC1234"

if __name__ == "__main__":
    img_path = "image.png"
    plate_text = extract_number_plate(img_path)
    print("Extracted License Plate Text:", plate_text)