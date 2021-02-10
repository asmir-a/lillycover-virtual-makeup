import numpy as np
from matplotlib import pyplot as plt
import face_recognition
import cv2


img_bgr = cv2.imread('woman1.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

face_landmarks = face_recognition.face_landmarks(img_rgb)[0]

for face_part, points in face_landmarks.items():
    if face_part == 'right_eyebrow':
        i = 0
        for point in points:
            cv2.circle(img_rgb, point, radius=int(i * 0.8), color=(0, 0, 0), thickness = -1)
            i = i + 1
    else:
        for point in points:
            cv2.circle(img_rgb, point, radius=3, color=(0, 0, 0), thickness=-1)

plt.imshow(img_rgb)
plt.show()