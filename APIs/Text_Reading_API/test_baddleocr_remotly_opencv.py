import cv2
import os
import warnings
import logging
import time
import requests


warnings.filterwarnings("ignore", category=UserWarning)

os.environ["PADDLE_NO_GPU"] = "0"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"


logging.basicConfig(level=logging.ERROR)


url = "https://98ba-197-38-78-133.ngrok-free.app/paddleocr/"


camera_url = "https://192.168.0.2:8080/video"
cap = cv2.VideoCapture(camera_url)


if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    last_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        current_time = time.time()
        if current_time - last_time >= 10:
            # Convert frame to JPEG in memory
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                print("Error: Failed to encode frame.")
                continue

            files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
            try:
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

            last_time = current_time

        
        time.sleep(0.01)

    cap.release()
