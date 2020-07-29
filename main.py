#####pythoncode.py#####
import numpy as np
import sys, os
from fastapi import FastAPI, UploadFile, File
from starlette.requests import Request
import io
import cv2
import pytesseract
from pydantic import BaseModel

app = FastAPI()


class ImageType(BaseModel):
    url: str


def read_img(img):
    pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseractgit '
    # pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    text = pytesseract.image_to_string(img)
    return (text)


app = FastAPI()



@app.post('/predict/')
def prediction(request: Request, file: bytes = File(...)):
    try:

        if request.method == 'POST':
            image_stream = io.BytesIO(file)
            image_stream.seek(0)
            file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            label = read_img(frame)
            return label
        return 'No post request found'

    except AssertionError:
        assert 'Erro ao processar'
