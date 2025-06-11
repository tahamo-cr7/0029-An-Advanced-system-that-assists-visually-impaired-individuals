
import cv2
import numpy as np
import time
import os
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import mediapipe as mp
import RPi.GPIO as GPIO  # For joystick button

# ------------------- TTS Function -------------------
def speak(text):
    print("Speaking:", text)
    try:
        obj = gTTS(text=text, lang='en', slow=False)
        obj.save("temp.mp3")
        play(AudioSegment.from_mp3("temp.mp3"))
    except Exception:
        print("gTTS failed, using espeak instead.")
        os.system(f'espeak \"{text}\"')

# ------------------- GPIO Setup -------------------
SW_PIN = 17 # GPIO pin connected to joystick's SEL button
GPIO.setmode(GPIO.BCM)
GPIO.setup(SW_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# ------------------- Load Models -------------------
face_interpreter = Interpreter(model_path="/home/taha/grad_project/face_recognition_model.tflite")
face_interpreter.allocate_tensors()
face_input_details = face_interpreter.get_input_details()
face_output_details = face_interpreter.get_output_details()

emotion_interpreter = Interpreter(model_path="/home/taha/grad_project/emotion_detection_model.tflite")
emotion_interpreter.allocate_tensors()
emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()

# ------------------- Labels -------------------
labels_dict = {
    0: "Nada", 1: "Nooran", 2: "Cristiano Ronaldo",
    3: "Leonardo DiCaprio", 4: "Magdy Yacoub", 5: "Mohamed Salah"
}
emotions_dict = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear',
    3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'
}

# ------------------- Setup PiCamera -------------------
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()
picam2.set_controls({"AfMode": 2})

# ------------------- MediaPipe Face Detection -------------------
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)


print("Face + Emotion Recognition is running... Press Joystick once to analyze, long press to quit.")
speak("Face and emotion recognition ready. Press the switch once to capture. or long press to quit.")

# ------------------- Main Loop -------------------
while True:
    frame = picam2.capture_array()
    display_frame = frame.copy()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("Face + Emotion Recognition", cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

    sw = GPIO.input(SW_PIN)

    if sw == GPIO.LOW:
        press_start_time = time.time()
        while GPIO.input(SW_PIN) == GPIO.LOW:
            time.sleep(0.01)
        press_duration = time.time() - press_start_time

        if press_duration >= 1:  # Long press to quit
            speak("Exiting.")
            break
        else:
            print("[INFO] Capturing and analyzing...")
            speak("Image is captured")
            results = face_detection.process(rgb_frame)
            if not results.detections:
                text = "No faces detected."
            else:
                spoken_sentences = []
                image_height, image_width, _ = frame.shape

                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * image_width)
                    y1 = int(bbox.ymin * image_height)
                    w = int(bbox.width * image_width)
                    h = int(bbox.height * image_height)
                    x2 = x1 + w
                    y2 = y1 + h

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(image_width, x2), min(image_height, y2)

                    sub_face_img = frame[y1:y2, x1:x2]
                    if sub_face_img.size == 0:
                        continue

                    resized_face = cv2.resize(sub_face_img, (224, 224))
                    rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
                    normalized_face = rgb_face.astype("float32") / 255.0
                    input_data_face = np.expand_dims(normalized_face, axis=0)

                    # Face recognition
                    face_interpreter.set_tensor(face_input_details[0]['index'], input_data_face)
                    face_interpreter.invoke()
                    face_result = face_interpreter.get_tensor(face_output_details[0]['index'])
                    face_label = np.argmax(face_result)
                    face_confidence = np.max(face_result)
                    face_name = labels_dict.get(face_label, "someone I don't recognize") if face_confidence >= 0.7 else "someone I don't recognize"

                    # Emotion recognition
                    emotion_interpreter.set_tensor(emotion_input_details[0]['index'], input_data_face)
                    emotion_interpreter.invoke()
                    emotion_result = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])
                    emotion_label = np.argmax(emotion_result)
                    emotion_name = emotions_dict.get(emotion_label, "Unknown")

                    label_text = f"{face_name} - {emotion_name}"
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    spoken_sentences.append(f"{face_name} looks {emotion_name}")

                text = ". ".join(spoken_sentences)

            print("[SPEAKING]:", text)
            speak(text)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break  # Optional emergency exit

# ------------------- Cleanup -------------------
cv2.destroyAllWindows()
GPIO.cleanup()
