import cv2
import numpy as np
from matplotlib import pyplot as plt
import face_recognition
from scipy import interpolate
from pylab import *

def get_discrete_points_left_eye(img_rgb):
    face_landmarks = face_recognition.face_landmarks(img_rgb)[0]
    points_eyeshadow_left = []

    # points_eyeshadow_left.append(face_landmarks['left_eye'][3])
    points_eyeshadow_left.append(face_landmarks['left_eye'][2])
    points_eyeshadow_left.append(face_landmarks['left_eye'][1])
    points_eyeshadow_left.append(face_landmarks['left_eye'][0])

    point0 = int((face_landmarks['chin'][0][0] + face_landmarks['left_eye'][0][0]) / 2), int((face_landmarks['chin'][0][1] + face_landmarks['left_eye'][0][1]) / 2)
    point1 = int(face_landmarks['left_eyebrow'][0][0] - point0[0]), int(face_landmarks['left_eyebrow'][0][1] - point0[1])
    point2 = int(point0[0] + point1[0] / 3), int(point0[1] + point1[1] / 3)
    points_eyeshadow_left.append(list(point2))
    # TODO: change the divisor: put something else instead of 3, but it actually looks quite fine

    point3 = int((face_landmarks['left_eyebrow'][0][0] + face_landmarks['left_eyebrow'][1][0]) / 2), int((face_landmarks['left_eyebrow'][0][1] + face_landmarks['left_eyebrow'][1][1]) / 2)
    point4 = int((face_landmarks['left_eye'][0][0] + point3[0]) / 2), int((face_landmarks['left_eye'][0][1] + point3[1]) / 2)
    points_eyeshadow_left.append(list(point4))
    # Looks good

    point5 = int((face_landmarks['left_eyebrow'][1][0] + face_landmarks['left_eyebrow'][2][0]) / 2), int((face_landmarks['left_eyebrow'][1][1] + face_landmarks['left_eyebrow'][2][1]) / 2)
    point6 = int((face_landmarks['left_eye'][1][0] + point5[0]) / 2), int((face_landmarks['left_eye'][1][1] + point5[1]) / 2)
    points_eyeshadow_left.append(list(point6))
    # Looks good, but can be a little lower

    point7 = int((face_landmarks['left_eyebrow'][3][0] + face_landmarks['left_eye'][1][0]) / 2), int((face_landmarks['left_eyebrow'][3][1] + face_landmarks['left_eye'][1][1]) / 2)
    points_eyeshadow_left.append(list(point7))
    # Looks good

    point8 = int(face_landmarks['left_eye'][3][0] - face_landmarks['left_eyebrow'][3][0]), int(face_landmarks['left_eye'][3][1] - face_landmarks['left_eyebrow'][3][1])
    point9 = int(face_landmarks['left_eyebrow'][3][0] + point8[0] * 1.5 / 3), int(face_landmarks['left_eyebrow'][3][1] + point8[1] * 1.5 / 3)
    points_eyeshadow_left.append(list(point9))
    points_eyeshadow_left.append(face_landmarks['left_eye'][2])
    # TODO: lower the point9

    points_eyeshadow_left = np.array(points_eyeshadow_left, dtype=np.uint32)
    return points_eyeshadow_left

def get_discrete_points_right_eye(img_rgb):
    face_landmarks = face_recognition.face_landmarks(img_rgb)[0]
    points_eyeshadow_right = []

    # points_eyeshadow_left.append(face_landmarks['left_eye'][0])
    points_eyeshadow_right.append(face_landmarks['right_eye'][1])
    points_eyeshadow_right.append(face_landmarks['right_eye'][2])
    points_eyeshadow_right.append(face_landmarks['right_eye'][3])

    point0 = int((face_landmarks['chin'][16][0] + face_landmarks['right_eye'][3][0]) / 2), int((face_landmarks['chin'][16][1] + face_landmarks['right_eye'][3][1]) / 2)
    point1 = int(face_landmarks['right_eyebrow'][4][0] - point0[0]), int(face_landmarks['right_eyebrow'][4][1] - point0[1])
    point2 = int(point0[0] + point1[0] / 3), int(point0[1] + point1[1] / 3)
    points_eyeshadow_right.append(list(point2))
    # TODO: change the divisor: put something else instead of 3, but it actually looks quite fine

    point3 = int((face_landmarks['right_eyebrow'][4][0] + face_landmarks['right_eyebrow'][3][0]) / 2), int((face_landmarks['right_eyebrow'][4][1] + face_landmarks['right_eyebrow'][3][1]) / 2)
    point4 = int((face_landmarks['right_eye'][3][0] + point3[0]) / 2), int((face_landmarks['right_eye'][3][1] + point3[1]) / 2)
    points_eyeshadow_right.append(list(point4))
    # Looks good

    point5 = int((face_landmarks['right_eyebrow'][3][0] + face_landmarks['right_eyebrow'][2][0]) / 2), int((face_landmarks['right_eyebrow'][3][1] + face_landmarks['right_eyebrow'][2][1]) / 2)
    point6 = int((face_landmarks['right_eye'][2][0] + point5[0]) / 2), int((face_landmarks['right_eye'][2][1] + point5[1]) / 2)
    points_eyeshadow_right.append(list(point6))
    # Looks good, but can be a little lower

    point7 = int((face_landmarks['right_eyebrow'][1][0] + face_landmarks['right_eye'][2][0]) / 2), int((face_landmarks['right_eyebrow'][1][1] + face_landmarks['right_eye'][2][1]) / 2)
    points_eyeshadow_right.append(list(point7))
    # Looks good

    point8 = int(face_landmarks['right_eye'][0][0] - face_landmarks['right_eyebrow'][1][0]), int(face_landmarks['right_eye'][0][1] - face_landmarks['right_eyebrow'][1][1])
    point9 = int(face_landmarks['right_eyebrow'][1][0] + point8[0] * 1.5 / 3), int(face_landmarks['right_eyebrow'][1][1] + point8[1] * 1.5 / 3)
    points_eyeshadow_right.append(list(point9))
    points_eyeshadow_right.append(face_landmarks['right_eye'][1])
    # TODO: lower the point9

    points_eyeshadow_right = np.array(points_eyeshadow_right, dtype=np.uint32)
    return points_eyeshadow_right

def get_exterior_points(points_eyeshadow_left):
    points_eyeshadow_left_x = points_eyeshadow_left[:, 0]
    points_eyeshadow_left_y = points_eyeshadow_left[:, 1]
    tck, u = interpolate.splprep([points_eyeshadow_left_x, points_eyeshadow_left_y], s=0, per=True)
    xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)
    xi = xi.astype(int)
    yi = yi.astype(int)
    points_eyeshadow = [tuple(elem) for elem in np.column_stack((xi, yi))]
    points_eyeshadow_sorted = sorted(set(points_eyeshadow))
    x_range_eyeshadow = np.linspace(points_eyeshadow_sorted[0][0],points_eyeshadow_sorted[len(points_eyeshadow_sorted) - 1][0], num=1001,endpoint=True)
    points_eyeshadow_sorted.pop(0)
    points_eyeshadow_sorted.pop(len(points_eyeshadow_sorted) - 1)
    points_eyeshadow_list = np.array(points_eyeshadow_sorted)
    return points_eyeshadow_list
    # TODO: this method does not seem to be safe: use fillPoly() somehow because algorithm may end up having just one point

def get_interior_points(x, y):
    intx = []
    inty = []

    def ext(a, b, i):
        a, b = round(a), round(b)
        intx.extend(arange(a, b, 1).tolist())
        inty.extend((ones(b - a) * i).tolist())

    x, y = np.array(x), np.array(y)
    xmin, xmax = amin(x), amax(x)
    xrang = np.arange(xmin, xmax + 1, 1)
    for i in xrang:
        ylist = y[where(x == i)]
        ext(amin(ylist), amax(ylist), i)
    return np.array(inty, dtype=np.int32), np.array(intx, dtype=np.int32)

def change_color(img_rgb, r = 128, g = 0, b = 128, intensity = 0.8):

    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    img_color = np.zeros(img_rgb.shape).astype('uint8')

    for i in range(img_color.shape[0]):
        for j in range(img_color.shape[1]):
            img_color[i, j, 0] = r
            img_color[i, j, 1] = g
            img_color[i, j, 2] = b
    img_color_lab = cv2.cvtColor(img_color, cv2.COLOR_RGB2LAB)

    l_mean_img_org = img_lab[:, :, 0].flatten().mean()
    a_mean_img_org = img_lab[:, :, 1].flatten().mean()
    b_mean_img_org = img_lab[:, :, 2].flatten().mean()

    l_color = img_color_lab[0, 0, 0]
    a_color = img_color_lab[0, 0, 1]
    b_color = img_color_lab[0, 0, 2]

    img_lab_shifted = img_lab + ((l_color - l_mean_img_org) * intensity, (a_color - a_mean_img_org) * intensity, (b_color - b_mean_img_org) * intensity)
    img_lab_shifted = np.clip(img_lab_shifted, 0, 255).astype('uint8')

    img_output_rgb = cv2.cvtColor(img_lab_shifted, cv2.COLOR_LAB2RGB)
    return img_output_rgb

def smoothen_blush(img_color_changed, img_original, points_x, points_y):
    height, width = img_color_changed.shape[:2]
    img_zeros = np.zeros((height, width))
    cv2.fillConvexPoly(img_zeros, np.array(np.c_[points_x, points_y], dtype = 'int32'), 1)
    img_mask_blur = cv2.GaussianBlur(img_zeros, (15, 15), 0)
    img_mask_blur_3d = np.ndarray([height, width, 3], dtype = 'float')
    img_mask_blur_3d[:, :, 0] = img_mask_blur
    img_mask_blur_3d[:, :, 1] = img_mask_blur
    img_mask_blur_3d[:, :, 2] = img_mask_blur
    img_out = (img_mask_blur_3d * img_color_changed + (1 - img_mask_blur_3d) * img_original).astype('uint8')
    return img_out

def apply_eyeshadow(img_rgb):
    points_eyeshadow_left = get_discrete_points_left_eye(img_rgb)
    points_eyeshadow_right = get_discrete_points_right_eye(img_rgb)

    points_eyeshadow_list_left = get_exterior_points(points_eyeshadow_left)
    points_eyeshadow_list_right = get_exterior_points(points_eyeshadow_right)

    x_interior_left, y_interior_left = get_interior_points(points_eyeshadow_list_left[:, 0],points_eyeshadow_list_left[:, 1])
    x_interior_right, y_interior_right = get_interior_points(points_eyeshadow_list_right[:, 0],points_eyeshadow_list_right[:, 1])

    img_mask_colored = change_color(img_rgb, 255, 0, 0, 0.3)
    img_output_left_only = smoothen_blush(img_mask_colored, img_rgb, x_interior_left, y_interior_left)
    img_output_left_and_right = smoothen_blush(img_mask_colored, img_output_left_only, x_interior_right,y_interior_right)

    return img_output_left_and_right

img_bgr = cv2.imread('output2.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

img_with_eyeshadow = apply_eyeshadow(img_rgb)

cv2.imwrite('output3.jpg', cv2.cvtColor(img_with_eyeshadow, cv2.COLOR_RGB2BGR))

plt.imshow(img_rgb)
plt.show()
plt.imshow(img_with_eyeshadow)
plt.show()