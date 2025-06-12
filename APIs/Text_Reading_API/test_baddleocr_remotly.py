import requests
import os
import warnings
import logging

# Suppress warnings and logs
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PADDLE_NO_GPU"] = "0"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
logging.basicConfig(level=logging.ERROR)


image_path = "C:/Users/Taha/Desktop/1.jpg"


if not os.path.exists(image_path):
    print("Error: Could not load image. Check the path.")
else:
    with open(image_path, "rb") as f:
        files = {"file": f}
        try:
            url = "https://c74d-197-38-78-133.ngrok-free.app/paddleocr/"
            response = requests.post(url, files=files)
            response.raise_for_status()

            data = response.json()

            if "detected_texts" in data and data["detected_texts"]:
                print("OCR result found:")
                for text in data["detected_texts"]:
                    print(f"Detected text: {text}")
            else:
                print("No text detected in the image.")

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
