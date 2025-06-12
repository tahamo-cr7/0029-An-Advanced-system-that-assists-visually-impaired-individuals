import cv2
import requests
import numpy as np

# Server URL
url = "https://4c2c-105-206-16-220.ngrok-free.app/caption/"
address = "http://192.168.142.64:8080/video"
cam = cv2.VideoCapture(address)

if not cam.isOpened():
    print("Cannot open camera stream")
    exit()

print("Starting captioning... Press Ctrl+C to stop")

prev_hist = None
similarity_threshold = 0.98  # Higher = more strict (e.g. 0.99 = very strict)

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        # Convert to grayscale and resize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (320, 240))

        # Calculate histogram and normalize
        hist = cv2.calcHist([gray_resized], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # Compare with previous histogram
        if prev_hist is not None:
            similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if similarity > similarity_threshold:
                continue  # Skip similar frame

        prev_hist = hist

        # Save frame
        image_path = "captured.jpg"
        cv2.imwrite(image_path, frame)

        # Send to server
        with open(image_path, "rb") as img:
            files = {"file": img}
            try:
                response = requests.post(url, files=files)
                caption = response.json().get("caption", "No caption returned")
                print("Caption:", caption)
            except Exception as e:
                print("Request failed:", e)

except KeyboardInterrupt:
    print("Captioning stopped by user")

cam.release()