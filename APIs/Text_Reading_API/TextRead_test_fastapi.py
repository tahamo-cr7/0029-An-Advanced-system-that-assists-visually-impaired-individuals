from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import logging
import warnings
from paddleocr import PaddleOCR

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PADDLE_NO_GPU"] = "0"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
logging.basicConfig(level=logging.ERROR)

app = FastAPI()
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

@app.post("/paddleocr/")
async def paddleocr(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        result = ocr.ocr(image, cls=True)

        if not result or result[0] is None:
            return JSONResponse(content={"message": "No text detected."}, status_code=200)

        detected_texts = []
        for line in result[0]:
            box = line[0]
            text = line[1][0]
            detected_texts.append({"text": text, "box": box})

        return {"detected_texts": detected_texts}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)