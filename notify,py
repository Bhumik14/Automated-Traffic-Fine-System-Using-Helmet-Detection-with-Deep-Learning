import json
import re
import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

# Load database
with open("./database/demo.json") as f:
    config = json.load(f)

ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.environ.get("TWILIO_PHONE")  # Twilio number

client = Client(ACCOUNT_SID, AUTH_TOKEN)

def preprocess_plate(plate: str) -> str:
    """Clean and normalize plate text."""
    if not plate:
        return ""
    # Remove spaces & special characters
    cleaned = re.sub(r"[^A-Za-z0-9]", "", plate)
    return cleaned.upper()

def send_notification(plate_text: str):
    """Send SMS notification for detected plate."""
    if plate_text:
        processed_plate = preprocess_plate(plate_text)

        # Lookup user info by normalized plate
        user_info = next(
            (item for item in config if preprocess_plate(item["number_plate_number"]) == processed_plate),
            None
        )

        if user_info:
            print(f"✅ Detected license plate: {processed_plate}")
            print("User Information:")
            for key, value in user_info.items():
                print(f"  {key}: {value}")

            # SMS body
            sms_body = (
                f"Dear {user_info['name']},\n"
                f"Your vehicle with plate {processed_plate} has been fined.\n"
                f"Address: {user_info['address']}.\n"

                
                f"Please pay the fine at the earliest."
            )

            # Send SMS if phone number exists
            phone = user_info.get("phone_number")
            if phone:
                try:
                    message = client.messages.create(
                        body=sms_body,
                        from_=TWILIO_PHONE,
                        to=f"+91{phone}"  # Assuming Indian numbers (add country code!)
                    )
                    print(f"📩 SMS sent successfully! SID: {message.sid}")
                except Exception as e:
                    print(f"❌ Failed to send SMS: {e}")
            else:
                print("⚠️ No phone number found for this user.")
        else:
            print(f"⚠️ License plate {processed_plate} not found in database.")
    else:
        print("⚠️ No license plate detected.")