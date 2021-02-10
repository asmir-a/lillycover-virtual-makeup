import face_recognition
import cv2 as cv
from matplotlib import pyplot as plt
import face_recognition
import numpy as np


def min_y_lip(bottom_lip_array):
    min_y = 0
    for key, value in bottom_lip_array:
        if value > min_y:
            min_y = value
    return min_y


def max_y_lip(top_lip_array):
    max_y = 10000
    for key, value in top_lip_array:
        if value < max_y:
            max_y = value
    return max_y


def left_x_lip(lip_array):
    min_x = 10000
    for key, value in lip_array:
        if key < min_x:
            min_x = key
    return min_x


def right_x_lip(lip_array):
    max_x = 0
    for key, value in lip_array:
        if key > max_x:
            max_x = key
    return max_x


def boundary_points(bottom_lip_array, top_lip_array):
    bot_y = min_y_lip(bottom_lip_array)
    top_y = max_y_lip(top_lip_array)
    left_x = left_x_lip(bottom_lip_array)
    right_x = right_x_lip(bottom_lip_array)
    return bot_y, top_y, left_x, right_x


image = cv.imread('woman1.jpg')
imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)

face_landmarks_list = face_recognition.face_landmarks(imageRGB)
face_landmarks = face_landmarks_list[0]

bottom_lip = face_landmarks["bottom_lip"]
top_lip = face_landmarks["top_lip"]
bottom_lip_len = len(bottom_lip)
print(top_lip)

lowest_point_lip, highest_point_lip, left_point_lip, right_point_lip = boundary_points(bottom_lip, top_lip)
print("The lowest lip coordinate is: {} {} {} {}".format(lowest_point_lip, highest_point_lip, left_point_lip, right_point_lip))
cv.rectangle(imageRGB, (left_point_lip - 2, highest_point_lip - 2), (right_point_lip + 2, lowest_point_lip + 2), (0, 0, 0))


#Identifying contours: works poorly on lips
imageGREY = cv.cvtColor(imageRGB, cv.COLOR_RGB2GRAY)
ret, thresh = cv.threshold(imageGREY, 80, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(imageGREY, contours, -1, (0, 255, 0), 1)


height, width, depth = imageRGB.shape

for i in range(highest_point_lip, lowest_point_lip):
    for j in range(left_point_lip, right_point_lip):
        if imageRGB[i, j, 1] < 170 and imageRGB[i, j, 2] < 170:
            imageRGB[i, j, 0] = 128
            imageRGB[i, j, 1] = 0
            imageRGB[i, j, 2] = 128

plt.imshow(imageRGB)
plt.show()





