import cv2
import numpy as np
from matplotlib import pyplot as plt
import face_recognition

img = cv2.imread('woman3.jpg')
img_yuv_present = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)


face_landmarks = face_recognition.face_landmarks(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
face_landmark = face_landmarks[0]

lip_bottom = face_landmark['bottom_lip']
lip_top = face_landmark['top_lip']
lip_points = lip_bottom + lip_top

print(lip_points)

poly_points = np.asarray(lip_points, dtype = np.int32)
poly_points = poly_points.reshape((-1, 1, 2))

cv2.fillPoly(img, [poly_points], (0, 0, 0))

img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)


shape = img_yuv.shape
print(shape)

for i in range(shape[0]):#looks good for the pink colored pomade
    for j in range(shape[1]):
        if img[i, j, 0] == 0 and img[i, j, 1] == 0 and img[i, j, 2] == 0:
            img_yuv_present[i, j, 1] += 30
            img_yuv_present[i, j, 2] += 30



img_present = cv2.cvtColor(img_yuv_present, cv2.COLOR_YUV2BGR)
cv2.imwrite('output1.jpg', img_present)
