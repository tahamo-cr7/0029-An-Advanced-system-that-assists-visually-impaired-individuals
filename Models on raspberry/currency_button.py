import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import RPi.GPIO as GPIO  # For joystick input

# ------------------- Setup GPIO -------------------
SW_PIN = 17 # Replace with your joystick button GPIO pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(SW_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# ------------------- CNN class names -------------------
class_names = ["5 EGP", "10 EGP", "10 EGP new", "20 EGP", "20 EGP new", "50 EGP", "100 EGP", "200 EGP"]

# ------------------- Speak function -------------------
def speak(text):
    print("Speaking:", text)
    try:
        obj = gTTS(text=text, lang='en', slow=False)
        obj.save("temp.mp3")
        play(AudioSegment.from_mp3("temp.mp3"))
    except Exception as e:
        print("gTTS failed, using espeak:", e)
        os.system(f'espeak \"{text}\"')

# ------------------- Load CNN model -------------------
cnn_interpreter = tflite.Interpreter(model_path="/home/taha/grad_project/converted_model.tflite")
cnn_interpreter.allocate_tensors()
input_details = cnn_interpreter.get_input_details()
output_details = cnn_interpreter.get_output_details()
input_shape = input_details[0]['shape']

# ------------------- Load YOLO model -------------------
yolo_model = YOLO("/home/taha/grad_project/best_ncnn_model")
yolo_labels = yolo_model.names

# ------------------- Setup PiCamera -------------------
resW, resH = 640, 480
picam = Picamera2()
picam.configure(picam.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
picam.start()

# Autofocus for Pi Camera V3
time.sleep(1)
picam.set_controls({"AfMode": 2})
time.sleep(0.1)
picam.set_controls({"AfTrigger": 1})
time.sleep(2)

print("[INFO] Press joystick to capture. Long press to quit.")
speak("Press the switch to capture image. Long press (one second) to quit.")

# ------------------- Main Loop -------------------
while True:
    frame = picam.capture_array()
    display_frame = cv2.resize(frame, (resW, resH))
    cv2.imshow("Live Preview", display_frame)

    sw = GPIO.input(SW_PIN)

    if sw == GPIO.LOW:
        press_start_time = time.time()
        while GPIO.input(SW_PIN) == GPIO.LOW:
            time.sleep(0.01)
        press_duration = time.time() - press_start_time

        if press_duration >= 1:
            speak("Exiting")
            break

        print("[INFO] Capturing image and running detection...")
        speak("Button pressed. Running detection.")
        results = yolo_model(display_frame, verbose=False)
        detections = results[0].boxes
        spoken_classes = set()

        for det in detections:
            xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            conf = det.conf.item()
            class_id = int(det.cls.item())
            label = yolo_labels[class_id]

            if conf > 0.4:
                roi = display_frame[ymin:ymax, xmin:xmax]
                if roi.size == 0:
                    continue

                cnn_img = cv2.resize(roi, (input_shape[2], input_shape[1])).astype(np.float32) / 255.0
                cnn_img = np.expand_dims(cnn_img, axis=0)

                cnn_interpreter.set_tensor(input_details[0]['index'], cnn_img)
                cnn_interpreter.invoke()
                output_data = cnn_interpreter.get_tensor(output_details[0]['index'])[0]

                cnn_class = np.argmax(output_data)
                cnn_conf = np.max(output_data)

                color = (0, 255, 0)
                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), color, 2)

                if cnn_conf > 0.6:
                    if cnn_class < len(class_names):
                        label_text = f"{label} ({int(conf*100)}%) -> {class_names[cnn_class]} ({int(cnn_conf*100)}%)"
                        if class_names[cnn_class] not in spoken_classes:
                            speak(class_names[cnn_class])
                            spoken_classes.add(class_names[cnn_class])
                    else:
                        label_text = f"{label} ({int(conf*100)}%) -> Unknown"
                        if "Unknown" not in spoken_classes:
                            speak("Unknown")
                            spoken_classes.add("Unknown")
                else:
                    label_text = f"{label} ({int(conf*100)}%) -> Low CNN Conf"
                    speak(label_text)

                cv2.putText(display_frame, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show result for 3 seconds then return to live feed
        cv2.imshow("Detection Result", display_frame)
        cv2.waitKey(1)
        time.sleep(3)
        cv2.destroyWindow("Detection Result")

    if cv2.waitKey(1) & 0xFF == ord('x'):  # Optional emergency exit
        break

# ------------------- Cleanup -------------------
picam.stop()
cv2.destroyAllWindows()
GPIO.cleanup()
