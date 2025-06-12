import cv2
import os
import warnings
import logging
from paddleocr import PaddleOCR

# Suppress warnings and logs
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PADDLE_NO_GPU"] = "0"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
logging.basicConfig(level=logging.ERROR)

# Initialize OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

# Load the image (Replace with your actual path)
image_path = "C:/Users/Taha/Desktop/text4.jpg"  # <--- CHANGE THIS
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image. Check the path.")
else:
    result = ocr.ocr(image, cls=True)

    if result is None or len(result) == 0 or result[0] is None:
        print("No text detected in the image.")
    else:
        print("OCR result found:")
        for line in result[0]:
            print(f"Detected text: {line[1][0]}")
