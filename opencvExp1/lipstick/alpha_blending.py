import cv2
import numpy as np
from matplotlib import pyplot as plt
import face_recognition


src1 = cv2.imread('woman1.jpg')
src2 = cv2.imread('woman1.jpg')

for i in range(src1.shape[0]):
    for j in range(src1.shape[1]):
        src2[i, j, 0] = 146
        src2[i, j, 1] = 161
        src2[i, j, 2] = 246

dst = cv2.addWeighted(src1, 0.2, src2, 0.5 ,0)

face_landmarks = face_recognition.face_landmarks(cv2.cvtColor(src1, cv2.COLOR_BGR2RGB))
face_landmark = face_landmarks[0]

top_lip = face_landmark['top_lip']
bottom_lip = face_landmark['bottom_lip']
poly_points = top_lip + bottom_lip
poly_points = np.asarray(poly_points, dtype=np.int32)
poly_points = poly_points.reshape((-1, 1, 2))

cv2.fillPoly(src1, [poly_points], (0, 0, 0))

for i in range(src1.shape[0]):
    for j in range(src1.shape[1]):
        if src1[i, j, 0] == 0 and src1[i, j, 1] == 0 and src1[i, j, 2] == 0:
            src1[i, j, 0] = dst[i, j, 0]
            src1[i, j, 1] = dst[i, j, 1]
            src1[i, j, 2] = dst[i, j, 2]


src2_rgb = cv2.cvtColor(src2, cv2.COLOR_BGR2RGB)
dst_rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
plt.imshow(cv2.cvtColor(src1, cv2.COLOR_BGR2RGB))
plt.show()
