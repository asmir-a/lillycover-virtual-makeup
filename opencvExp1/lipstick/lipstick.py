import cv2
import numpy as np
from matplotlib import pyplot as plt
import face_recognition

def image_stats(image):
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())
    return (lMean, lStd, aMean, aStd, bMean, bStd)


img = cv2.imread('woman1.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

face_landmarks = face_recognition.face_landmarks(img_rgb)
face_landmark = face_landmarks[0]

top_lip = face_landmark['top_lip']
bottom_lip = face_landmark['bottom_lip']
poly_points = top_lip + bottom_lip

print(top_lip)
print(bottom_lip)
print(poly_points)

poly_points = np.asarray(poly_points, dtype=np.int32)
poly_points = poly_points.reshape((-1, 1, 2))

cv2.fillPoly(img_rgb, [poly_points], (255, 0, 0))

img_red = np.full(img.shape, 0)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img_red[i, j, 0] = 128
        img_red[i, j, 1] = 0
        img_red[i, j, 2] = 128


img_red = np.ones(img.shape, dtype = "uint8")
print(img_red)

for i in range(img_red.shape[0]):
    for j in range(img_red.shape[1]):
        img_red[i, j, 0] = 0
        img_red[i, j, 1] = 128
        img_red[i, j, 2] = 0
plt.imshow(img_red)
plt.show()

target = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype("float32")
source = cv2.cvtColor(img_red, cv2.COLOR_BGR2LAB).astype("float32")

(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

(l, a, b) = cv2.split(target)
l -= lMeanTar
a -= aMeanTar
b -= bMeanTar

l = (lStdTar / lStdSrc) * l
a = (aStdTar / aStdSrc) * a
b = (bStdTar / bStdSrc) * b

l += lMeanSrc
a += aMeanSrc
b += bMeanSrc

l = np.clip(l, 0, 255)
a = np.clip(a, 0, 255)
b = np.clip(b, 0, 255)

transfer = cv2.merge([l, a, b])
transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)



plt.imshow(transfer)
plt.show()

print(source)


