import cv2
import numpy
import imread


class Filters():

    def __init__(self):
        pass

    def gray(imagem):

        return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

