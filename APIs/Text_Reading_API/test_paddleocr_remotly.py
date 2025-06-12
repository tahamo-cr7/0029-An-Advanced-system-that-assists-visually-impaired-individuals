import requests
import os
import warnings
import logging
import cv2
import numpy as np
from gtts import gTTS

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PADDLE_NO_GPU"] = "0"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
logging.basicConfig(level=logging.ERROR)

image_path = "1.jpg"

if not os.path.exists(image_path):
    print("Error: Could not load image. Check the path.")
else:
    image = cv2.imread(image_path)

    with open(image_path, "rb") as f:
        files = {"file": f}
        try:
            url = "https://c74d-197-38-78-133.ngrok-free.app/paddleocr/"
            response = requests.post(url, files=files)
            response.raise_for_status()

            data = response.json()

            if "detected_texts" in data and data["detected_texts"]:
                texts = []
                for det in data["detected_texts"]:
                    text = det.get("text", "").strip()
                    box = det.get("box", [])

                    if box and len(box) == 4:
                        pts = [(int(x), int(y)) for x, y in box]
                        cv2.polylines(image, [np.array(pts)], isClosed=True, color=(0, 255, 0), thickness=2)
                        if text:
                            if not text.endswith("."):
                                text += "."
                            texts.append(text)

                # تجميع كل جملتين في سطر
                lines = []
                for i in range(0, len(texts), 2):
                    chunk = " ".join(texts[i:i+2])
                    lines.append(chunk)

                final_text = "\n".join(lines)

                print("\nFormatted OCR Output:\n")
                print(final_text)

                # تحويل إلى صوت mp3 فقط
                tts = gTTS(text=final_text, lang="en")
                tts.save("ocr_output.mp3")
                print("\n Saved as ocr_output.mp3")

            else:
                print("No text detected in the image.")

        except requests.exceptions.RequestException as e:
            print(f"\nRequest failed: {e}")

    cv2.imshow("Detected Text", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()