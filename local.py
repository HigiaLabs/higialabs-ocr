try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
from cv2 import *
import cv2
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

imagem = 'modelo_cupom_fiscal.png'

img = cv2.imread(imagem)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gauss = cv2.GaussianBlur(gray, (1,1), 5)

ret, thresh = cv2.threshold(gauss, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
texto = pytesseract.image_to_string(thresh,lang='por')

with open("texto.txt", "w") as text_file:
    text_file.write(texto)

cv2.imshow('Imagem', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()



