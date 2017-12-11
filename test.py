import numpy as np
import cv2

s_cascade = cv2.CascadeClassifier('data/haarcascade_profileface.xml')


img = cv2.imread('head5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



results = s_cascade.detectMultiScale(gray, 1.3, 5)
print("r", results)

for (x, y, w, h) in results:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()