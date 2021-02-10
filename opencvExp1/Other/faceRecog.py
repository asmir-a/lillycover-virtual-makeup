from cv2 import imread
from cv2 import CascadeClassifier
from cv2 import rectangle
from cv2 import cvtColor
from cv2 import COLOR_BGR2RGB
from matplotlib import pyplot as plt

pixels = imread('woman1.jpg')
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

bboxes = classifier.detectMultiScale(pixels)

for box in bboxes:
    x, y, width, height = box
    x2, y2 = x + width, y + height
    rectangle(pixels, (x,y), (x2,y2), (0,0,255), 1)
    #print(box)

RGB_pixels = cvtColor(pixels, COLOR_BGR2RGB)

plt.imshow(RGB_pixels)
plt.show()