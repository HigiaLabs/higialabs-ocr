#####pythoncode.py#####
import numpy as np
import sys, os
from fastapi import FastAPI, UploadFile, File
from pydantic.env_settings import BaseSettings
from starlette.requests import Request
import io
import cv2
import pytesseract
from pydantic import BaseModel

from utils.processing.face_detect import Face

app = FastAPI()


class Settings(BaseSettings):
    ...

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


settings = Settings(_env_file='.env', _env_file_encoding='utf-8')

print(settings)

class ImageType(BaseModel):
    url: str


def read_img(img):
    pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'
    text = pytesseract.image_to_string(img)
    return (text)


app = FastAPI()


@app.post('/ocr/')
def ocr(request: Request, file: bytes = File(...)):
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


@app.post('/face-detect/')
def face_detect(request: Request, file: bytes = File(...)):
    try:

        face = Face()

        image_stream = io.BytesIO(file)
        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # label = read_img(frame)
        num, data = face.detect(frame)

        content = {
            "faces": num,
            "imagem_cinza": data,

        }

        return content


    except AssertionError:
        assert 'Erro ao processar'
