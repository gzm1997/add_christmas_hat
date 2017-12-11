import numpy as np
from PIL import Image
import cv2

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor = 1.3, minNeighbors = 4, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects[0]

def draw_rects(img, rect, color):
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt.xml")
    nested = cv2.CascadeClassifier("data/haarcascade_eye.xml")

    img = Image.open('head3.jpg')
    hat = Image.open('hat2.png')
    heigh, width, d = (np.array(hat)).shape

    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    rect = detect(gray, cascade)
    x, y, w, h = rect
    vis = img.copy()

    new_heigh = int(heigh * w / width)
    hat = hat.resize((w, new_heigh), Image.ANTIALIAS)
    r, g, b, a = hat.split()
    vis.paste(hat, (x, y - new_heigh), mask = a)
    vis.show()


    img = Image.open('head5.jpg')
    hat = Image.open('hat2.png')
    heigh, width, d = (np.array(hat)).shape

    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    rect = detect(gray, cascade)
    x, y, w, h = rect
    vis = img.copy()

    new_heigh = int(heigh * w / width)
    hat = hat.resize((w, new_heigh), Image.ANTIALIAS)
    r, g, b, a = hat.split()
    vis.paste(hat, (x, y - new_heigh), mask = a)
    vis.show()