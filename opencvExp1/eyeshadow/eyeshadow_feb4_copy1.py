import cv2
import numpy as np
from matplotlib import pyplot as plt
import face_recognition
from scipy import interpolate
from pylab import *
from skimage import color


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
    return np.array(intx, dtype=np.int32), np.array(inty, dtype=np.int32)

def get_interior_points_1(x, y):
    intx = []
    inty = []
    def ext(a, b, i):
        a, b = round(a), round(b)
        intx.extend(arange(a, b, 1).tolist())
        inty.extend((ones(b-a) * i).tolist())
    x_min = np.amin(x)
    x_max = np.amax(x)
    x_range = np.arange(x_min, x_max + 1, 1)
    for i in x_range:
        y_list = y[where(x == i)]
        y_list = np.array(y_list)
        ext(np.amin(y_list), np.amax(y_list), i)
    return np.array(intx, dtype = np.int32), np.array(inty, dtype = np.int32)

def apply_blush(img_rgb, r = 128.0, g = 0.0, b = 128.0):
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    color_base = np.zeros(img_rgb.shape)
    for i in range(zeros.shape[0]):
        for j in range(zeros.shape[1]):
            zeros[i, j, :] = (r, g, b)
    color_base_lab = cv2.cvtColor(color_base, cv2.COLOR_RGB2LAB)

intensity = 1.0
def apply_blush_color(img, r=160., g=0., b=170.):
    height, width = img.shape[:2]
    val = color.rgb2lab((img / 255.)).reshape(width * height, 3)
    L, A, B = mean(val[:, 0]), mean(val[:, 1]), mean(val[:, 2])
    L1, A1, B1 = color.rgb2lab(np.array((r / 255., g / 255., b / 255.)).reshape(1, 1, 3)).reshape(3, )
    ll, aa, bb = (L1 - L) * intensity, (A1 - A) * intensity, (B1 - B) * intensity
    val[:, 0] = np.clip(val[:, 0] + ll, 0, 100)
    val[:, 1] = np.clip(val[:, 1] + aa, -127, 128)
    val[:, 2] = np.clip(val[:, 2] + bb, -127, 128)
    img = color.lab2rgb(val.reshape(height, width, 3)) * 255
    return img

def smoothen_blush(img, imgOrg, x1, y1):
    height, width = img.shape[:2]
    imgBase = zeros((height, width))
    cv2.fillConvexPoly(imgBase, np.array(c_[x1, y1], dtype='int32'), 1)
    imgMask = cv2.GaussianBlur(imgBase, (51, 51), 0)
    imgBlur3D = np.ndarray([height, width, 3], dtype='float')
    imgBlur3D[:, :, 0] = imgMask
    imgBlur3D[:, :, 1] = imgMask
    imgBlur3D[:, :, 2] = imgMask
    imgOrg = (imgBlur3D * img + (1 - imgBlur3D) * imgOrg).astype('uint8')
    return imgOrg

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

def smoothen_blush_my(img_color_changed, img_original, points_x, points_y):
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


img_bgr = cv2.imread('woman1.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


face_landmarks = face_recognition.face_landmarks(img_rgb)[0]

# for face_part, points in face_landmarks.items():
#     if face_part == 'left_eye' or face_part == 'left_eyebrow' or face_part == 'chin':
#         for point in points:
#             cv2.circle(img_rgb, point, radius=2, color=(0, 0, 0), thickness = -1)

points_eyeshadow_left = []
# points_eyeshadow_left.append(face_landmarks['left_eye'][3])
points_eyeshadow_left.append(face_landmarks['left_eye'][2])
points_eyeshadow_left.append(face_landmarks['left_eye'][1])
points_eyeshadow_left.append(face_landmarks['left_eye'][0])




point0 = int((face_landmarks['chin'][0][0] + face_landmarks['left_eye'][0][0]) / 2), int((face_landmarks['chin'][0][1] + face_landmarks['left_eye'][0][1]) / 2)
point1 = int(face_landmarks['left_eyebrow'][0][0] - point0[0]), int(face_landmarks['left_eyebrow'][0][1] - point0[1])
point2 = int(point0[0] + point1[0] / 3), int(point0[1] + point1[1] / 3)
points_eyeshadow_left.append(list(point2))
#TODO: change the divisor: put something else instead of 3, but it actually looks quite fine

point3 = int((face_landmarks['left_eyebrow'][0][0] + face_landmarks['left_eyebrow'][1][0]) / 2), int((face_landmarks['left_eyebrow'][0][1] + face_landmarks['left_eyebrow'][1][1]) / 2)
point4 = int((face_landmarks['left_eye'][0][0] + point3[0]) / 2), int((face_landmarks['left_eye'][0][1] + point3[1]) / 2)
points_eyeshadow_left.append(list(point4))
#Looks good

point5 = int((face_landmarks['left_eyebrow'][1][0] + face_landmarks['left_eyebrow'][2][0]) / 2), int((face_landmarks['left_eyebrow'][1][1] + face_landmarks['left_eyebrow'][2][1]) / 2)
point6 = int((face_landmarks['left_eye'][1][0] + point5[0]) / 2), int((face_landmarks['left_eye'][1][1] + point5[1]) / 2)
points_eyeshadow_left.append(list(point6))
#Looks good, but can be a little lower

point7 = int((face_landmarks['left_eyebrow'][3][0] + face_landmarks['left_eye'][1][0]) / 2), int((face_landmarks['left_eyebrow'][3][1] + face_landmarks['left_eye'][1][1]) / 2)
points_eyeshadow_left.append(list(point7))
# Looks good

point8 = int(face_landmarks['left_eye'][3][0] - face_landmarks['left_eyebrow'][3][0]), int(face_landmarks['left_eye'][3][1] - face_landmarks['left_eyebrow'][3][1])
point9 = int(face_landmarks['left_eyebrow'][3][0] + point8[0] * 1.5 / 3), int(face_landmarks['left_eyebrow'][3][1] + point8[1] * 1.5 / 3)
points_eyeshadow_left.append(list(point9))
points_eyeshadow_left.append(face_landmarks['left_eye'][2])
# TODO: lower the point9

points_eyeshadow_left = np.array(points_eyeshadow_left, dtype = np.uint32)
points_eyeshadow_left_x = points_eyeshadow_left[:, 0]
points_eyeshadow_left_y = points_eyeshadow_left[:, 1]

tck, u = interpolate.splprep([points_eyeshadow_left_x, points_eyeshadow_left_y], s = 0, per = True)
xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)

xi = xi.astype(int)
yi = yi.astype(int)

points_eyeshadow = [tuple(elem) for elem in np.column_stack((xi, yi))]
points_eyeshadow_sorted = sorted(set(points_eyeshadow))


x_range_eyeshadow = np.linspace(points_eyeshadow_sorted[0][0], points_eyeshadow_sorted[len(points_eyeshadow_sorted) - 1][0], num = 1001, endpoint = True)

points_eyeshadow_sorted.pop(0)
points_eyeshadow_sorted.pop(len(points_eyeshadow_sorted) - 1)
# points_eyeshadow_list = [list(elem) for elem in points_eyeshadow_sorted]
points_eyeshadow_tuple = [tuple(elem) for elem in points_eyeshadow_sorted]

points_eyeshadow_list = np.array(points_eyeshadow_sorted)

x_interior, y_interior = get_interior_points(points_eyeshadow_list[:, 0], points_eyeshadow_list[:, 1])
# TODO: this method does not seem to be safe: use fillPoly() somehow because algorithm may end up having just one point


# cv2.fillConvexPoly(img_rgb, np.array(points_eyeshadow, 'int32'), (0, 0, 0), 1)

x_interior_len = len(x_interior)

# for i in range(x_interior_len):
#     cv2.circle(img_rgb, (y_interior[i], x_interior[i]), radius = 1, color = (128, 0, 128), thickness = -1)




img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
color_base = np.zeros(img_rgb.shape).astype('uint8')
for i in range(color_base.shape[0]):
    for j in range(color_base.shape[1]):
        color_base[i, j, 0] = 128
        color_base[i, j, 1] = 0
        color_base[i, j, 2] = 128
color_base_lab = cv2.cvtColor(color_base, cv2.COLOR_RGB2LAB)

l_mean_img_orig = int(img_lab[:, :, 0].flatten().mean())
a_mean_img_orig = int(img_lab[:, :, 1].flatten().mean())
b_mean_img_orig = int(img_lab[:, :, 2].flatten().mean())

l_color = color_base_lab[0, 0, 0]
a_color = color_base_lab[0, 0, 1]
b_color = color_base_lab[0, 0, 2]
# Scalar operations work fine with numpy arrays. Therefore, it is not necessary to loop through all the pixel values of an image
img_lab = img_lab + (l_color - l_mean_img_orig, a_color - a_mean_img_orig, b_color - b_mean_img_orig)
img_lab = np.clip(img_lab, 1, 255)
#Opencv methods expect an image to be encoded in uint8 format. Therefore, when I use a int32 format, it thinks that there are moere dimensions than there are

#The result of the color change:
# img_output1 = cv2.cvtColor(np.uint8(img_lab), cv2.COLOR_LAB2RGB)

img_output1 = change_color(img_rgb, 128, 0, 128, 1)
img_original = img_rgb.copy()
img_output2 = smoothen_blush_my(img_output1, img_original, y_interior, x_interior)


img_lab1 = cv2.GaussianBlur(img_output1, (9, 9), 0)

rectangle_and_blur = np.zeros(img_lab1.shape)
print(points_eyeshadow_list)
rectangle_and_blur = np.uint8(rectangle_and_blur)
cv2.fillPoly(rectangle_and_blur, np.int32([points_eyeshadow_list]), (1, 1, 1), 1)
rectangle_and_blur_dot_product = rectangle_and_blur * img_lab1
rectangle_and_blur_mult = np.multiply(rectangle_and_blur, img_lab1)
rectangle_and_blur = rectangle_and_blur_mult

img_rgb_copy = img_rgb.copy()
img_without_rectangle = cv2.fillPoly(img_rgb_copy, np.int32([points_eyeshadow_list]), (0, 0, 0), 1)

img_eyeshadow_applied = img_without_rectangle + rectangle_and_blur

img_rgb_copy1 = img_rgb.copy()
img_rgb_copy2 = img_rgb.copy()
im = apply_blush_color(img_rgb_copy1)
imOrg = smoothen_blush(im, img_rgb_copy2, y_interior, x_interior)

cv2.circle(img_rgb, point2, radius = 1, color = (128, 0, 128), thickness = -1)
cv2.circle(img_rgb, point4, radius = 1, color = (128, 0, 128), thickness = -1)
cv2.circle(img_rgb, point6, radius = 1, color = (128, 0, 128), thickness = -1)
cv2.circle(img_rgb, point7, radius = 1, color = (128, 0, 128), thickness = -1)
cv2.circle(img_rgb, point9, radius = 1, color = (128, 0, 128), thickness = -1)
# cv2.fillPoly(img_rgb, points_eyeshadow_left, (128, 0, 128), 1)

plt.figure()
f, axarr = plt.subplots(1, 3)
axarr[0].imshow(img_output1)
axarr[1].imshow(img_rgb)
# axarr[2].imshow(rectangle_and_blur_dot_product)
# axarr[3].imshow(rectangle_and_blur)
# axarr[4].imshow(img_without_rectangle)
axarr[2].imshow(img_output2)
# plt.plot(y_interior, x_interior, 'o')
plt.show()