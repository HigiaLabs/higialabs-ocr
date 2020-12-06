import cv2
import numpy as np

class Draw():
    def __init__(self):
        pass

    def draw_rectangle(img, rect):
        '''Draw a rectangle on the image'''
        (x, y, w, h) = rect
        return cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)