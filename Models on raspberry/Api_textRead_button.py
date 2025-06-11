import cv2
import requests
import os
import time
import warnings
import logging
from picamera2 import Picamera2
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import RPi.GPIO as GPIO

# ------------------- Suppress Warnings -------------------
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PADDLE_NO_GPU"] = "0"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
logging.basicConfig(level=logging.ERROR)

# ------------------- GPIO Setup -------------------
BUTTON_GPIO = 17  # Change this if your button is on another GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# ------------------- TTS Function -------------------
def speak(text):
    print("Speaking:", text)
    try:
        obj = gTTS(text=text, lang='en', slow=False)
        obj.save("temp.mp3")
        play(AudioSegment.from_mp3("temp.mp3"))
    except Exception as e:
        print("gTTS failed, using espeak instead.")
        os.system(f'espeak \"{text}\"')

# ------------------- Initialize Camera -------------------
print("Initializing camera...")
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
picam2.start()
picam2.set_controls({"AfMode": 2})
time.sleep(1)

print("Joystick ready: Short press to capture, long press (= 1s) to quit.")
speak("Press the switch to capture image. Long press (1 second) to quit.")

# ------------------- OCR Capture Function -------------------
def capture_and_ocr(frame):
    image_path = "/home/taha/temp_ocr_image.jpg"
    cv2.imwrite(image_path, frame)
    print("Image captured and saved.")

    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            files = {"file": f}
            try:
                url = "https://7786-197-38-242-205.ngrok-free.app/paddleocr/"
                response = requests.post(url, files=files)
                response.raise_for_status()

                data = response.json()

                if "detected_texts" in data and data["detected_texts"]:
                    text_list = [item["text"] for item in data["detected_texts"] if "text" in item]

                    if text_list:
                        print("OCR result found:")
                        for text in text_list:
                            print(f"Detected text: {text}")
                        combined_text = " ".join(text_list)
                        speak(combined_text)
                    else:
                        print("No readable text found.")
                        speak("No readable text found.")
                else:
                    print("No text detected in the image.")
                    speak("No text detected.")
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
            finally:
                os.remove(image_path)
    else:
        print("Error: Could not save or find image.")

# ------------------- Main Loop -------------------
try:
    while True:
        frame = picam2.capture_array()
        cv2.imshow("Live Preview - Use Joystick", frame)
        cv2.waitKey(1)

        if GPIO.input(BUTTON_GPIO) == GPIO.LOW:
            press_time = time.time()
            while GPIO.input(BUTTON_GPIO) == GPIO.LOW:
                time.sleep(0.01)
            hold_duration = time.time() - press_time

            if hold_duration >= 1:
                print("[INFO] Long press detected. Exiting.")
                speak("Exiting.")
                break
            else:
                print("[INFO] Short press detected. Capturing image for OCR...")
                speak("Capturing image.")
                capture_and_ocr(frame)

except KeyboardInterrupt:
    pass
finally:
    print("[INFO] Cleaning up...")
    cv2.destroyAllWindows()
    picam2.close()
    GPIO.cleanup()