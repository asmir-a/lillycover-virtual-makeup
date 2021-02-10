import cv2
from matplotlib import pyplot as plt
import face_recognition
import numpy as np


img = cv2.imread('woman1.jpg')
img_hsv_final = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

face_landmarks = face_recognition.face_landmarks(img_rgb)
face_landmark = face_landmarks[0]

poly_points = face_landmark['top_lip'] + face_landmark['bottom_lip']

poly_points = np.asarray(poly_points, dtype = np.int32)
poly_points = poly_points.reshape((-1, 1, 2))

cv2.fillPoly(img_rgb, [poly_points], (0, 0, 0))

img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

for i in range(img_hsv.shape[0]):
    for j in range(img_hsv.shape[1]):
        if img_hsv[i, j, 0] == 0 and img_hsv[i, j, 1] == 0 and img_hsv[i, j, 2] == 0:
            print(img_hsv_final[i, j, 0])
            img_hsv_final[i, j, 0] = 30
            img_hsv_final[i, j, 1] += 10

img_rgb = cv2.cvtColor(img_hsv_final, cv2.COLOR_HSV2RGB)

plt.imshow(img_rgb)
plt.show()




