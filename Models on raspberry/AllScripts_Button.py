import subprocess
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import os
import spidev
import RPi.GPIO as GPIO
import time

# ---------- Speak Function ----------
def speak(text):
    print("Speaking:", text)
    try:
        obj = gTTS(text=text, lang='en', slow=False)
        obj.save("temp.mp3")
        play(AudioSegment.from_mp3("temp.mp3"))
    except Exception as e:
        print(f"gTTS failed, fallback to espeak: {e}")
        os.system(f'espeak "{text}"')

# ---------- Run Script ----------
def run_script(script_name):
    try:
        subprocess.run(["python3", script_name])
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        speak(f"Error running {script_name}")

# ---------- SPI + GPIO Setup ----------
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000

SW_PIN = 17

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SW_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

setup_gpio()

def read_channel(channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    return ((adc[1] & 3) << 8) + adc[2]

CENTER_MIN = 450
CENTER_MAX = 570

# ---------- Main ----------
speak("Hi, Welcome dear")
time.sleep(0.2)

last_click_time = 0
double_click_threshold = 0.6  # seconds
button_pressed = False

try:
    while True:
        print("\nPlease choose a mode using the joystick.")
        speak("Please choose a mode using the joystick. Move it Right for Text reading mode, move left for Currency mode, move up for Scene description mode, move down for face and emotion recognition mode, or double press to quit.")
        print("Choose now:")
        speak("Choose now")

        while True:
            x = read_channel(0)
            y = read_channel(1)
            sw = GPIO.input(SW_PIN)

            if x < CENTER_MIN and CENTER_MIN <= y <= CENTER_MAX:
                print("Left detected: Currency mode")
                speak("Left detected. Currency mode is chosen.")
                try:
                    run_script("/home/taha/grad_project/Scripts/Button_Scripts/currency_button.py")
                finally:
                    GPIO.cleanup()
                    time.sleep(0.2)
                    setup_gpio()
                break

            elif x > CENTER_MAX and CENTER_MIN <= y <= CENTER_MAX:
                print("Right detected: Text reading mode")
                speak("Right detected. Text reading mode is chosen.")
                try:
                    run_script("/home/taha/grad_project/Scripts/Button_Scripts/Api_textRead_button.py")
                finally:
                    GPIO.cleanup()
                    time.sleep(0.2)
                    setup_gpio()
                break

            elif y < CENTER_MIN and CENTER_MIN <= x <= CENTER_MAX:
                print("Up detected: Scene description mode")
                speak("Up detected. Scene description mode is chosen.")
                try:
                    run_script("/home/taha/grad_project/Scripts/Button_Scripts/api_scene_description_button.py")
                finally:
                    GPIO.cleanup()
                    time.sleep(0.2)
                    setup_gpio()
                break

            elif y > CENTER_MAX and CENTER_MIN <= x <= CENTER_MAX:
                print("Down detected: Face and emotion recognition mode")
                speak("Down detected. Face and emotion recognition mode is chosen.")
                try:
                    run_script("/home/taha/grad_project/Scripts/Button_Scripts/face_emotion_Button.py")
                finally:
                    GPIO.cleanup()
                    time.sleep(0.2)
                    setup_gpio()
                break

            # Double click detection
            if sw == GPIO.LOW and not button_pressed:
                button_pressed = True
                current_time = time.time()
                if current_time - last_click_time < double_click_threshold:
                    speak("Double click detected. Goodbye.")
                    raise KeyboardInterrupt
                last_click_time = current_time

            elif sw == GPIO.HIGH:
                button_pressed = False

            time.sleep(0.1)

except KeyboardInterrupt:
    print("Program exiting via double-click or interrupt.")
finally:
    spi.close()
    GPIO.cleanup()
