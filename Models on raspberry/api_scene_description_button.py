import cv2
import os
import requests
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from picamera2 import Picamera2
import time
import RPi.GPIO as GPIO

# ----------- GPIO Setup -----------
BUTTON_GPIO = 17  # Adjust this GPIO pin number if needed
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# ----------- Text-to-Speech Function -----------
def speak(text):
    print("Speaking:", text)
    try:
        obj = gTTS(text=text, lang='en', slow=False)
        obj.save("caption.mp3")
        play(AudioSegment.from_mp3("caption.mp3"))
    except Exception as e:
        print("gTTS failed, using espeak instead.")
        os.system(f'espeak \"{text}\"')

# ----------- Initialize PiCamera -----------
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
picam2.start()
picam2.set_controls({"AfMode": 2})
time.sleep(1)

# ----------- Server URL -----------
url = "https://7786-197-38-242-205.ngrok-free.app/caption/"

print("[INFO] Press switch to capture. Long press (1 second) to quit.")
speak("Press the switch to capture image. Long press (1 second) to quit.")

# ----------- Capture and Describe Function -----------
def capture_and_describe(frame):
    image_path = "/home/taha/temp_scene.jpg"
    cv2.imwrite(image_path, frame)
    print("[INFO] Image captured and saved.")

    try:
        with open(image_path, "rb") as img:
            files = {"file": img}
            response = requests.post(url, files=files)
            if response.status_code == 200:
                caption = response.json().get("caption", "No caption received.")
                print("Caption:", caption)
                speak(caption)
            else:
                print("Server error:", response.status_code)
                speak("Failed to get description from server.")
    except Exception as e:
        print("Error:", str(e))
        speak("Something went wrong.")

# ----------- Main Loop -----------
try:
    while True:
        frame = picam2.capture_array()
        cv2.imshow("Scene Description", frame)
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
                print("[INFO] Short press detected. Capturing image for description.")
                speak("Capturing image.")
                capture_and_describe(frame)

except KeyboardInterrupt:
    pass
finally:
    print("[INFO] Cleaning up...")
    cv2.destroyAllWindows()
    picam2.close()
    GPIO.cleanup()