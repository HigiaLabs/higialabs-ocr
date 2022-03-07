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


app = FastAPI(
    title="Higia Labs Vision",
    description="Projeto que implementa detecção de faces",
    version="1.0"
)


@app.post('/face-detect/')
def face_detect(request: Request, file: bytes = File(...)):
    try:

        face = Face()

        image_stream = io.BytesIO(file)
        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        num, data = face.detect(frame)
        content = {
            "num_faces": num,
        }
        return content
    except AssertionError:
        assert 'Erro ao processar'
