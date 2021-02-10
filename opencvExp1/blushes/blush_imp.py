import cv2
import numpy as np
import face_recognition
from scipy import interpolate
from matplotlib import pyplot as plt
from pylab import *
from skimage import color

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)

    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y

def get_boundary_points(x, y):
    tck, u = interpolate.splprep([x, y], s=0, per=1)
    unew = np.linspace(u.min(), u.max(), 1000)
    ynew, xnew = interpolate.splev(unew, tck, der=0)
    tup = c_[xnew.astype(int), ynew.astype(int)].tolist()
    coord = list(set(tuple(map(tuple, tup))))
    coord = np.array([list(elem) for elem in coord])
    return np.array(coord[:, 0], dtype=np.int32), np.array(coord[:, 1], dtype=np.int32)




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


# Note: probably, rgb color scheme is represented using values from 0 to 1 and the lab color scheme is represented using the values from 0 to 100 for l and from -127 to 128 for the a and b values
intensity = 0.4
def apply_blush_color(img, r=128., g=0., b=128.):
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

img_bgr = cv2.imread('output1.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

face_landmarks = face_recognition.face_landmarks(img_rgb)[0]
print(face_landmarks)

point_chin1 = face_landmarks['chin'][2]
point_chin2 = face_landmarks['chin'][3]
point_chin3 = face_landmarks['chin'][4]

point_lip1 = face_landmarks['top_lip'][0]

point_nose1 = face_landmarks['nose_bridge'][0]
point_nose2 = face_landmarks['nose_bridge'][3]
point_nose3 = face_landmarks['nose_tip'][0]

point_eye1 = face_landmarks['left_eye'][0]
point_eye2 = face_landmarks['left_eye'][3]

point_right_chin1 = face_landmarks['chin'][14]
point_right_chin2 = face_landmarks['chin'][13]
point_right_chin3 = face_landmarks['chin'][12]

point_right_lip1 = face_landmarks['top_lip'][6]

point_right_nose1 = face_landmarks['nose_bridge'][0]
point_right_nose2 = face_landmarks['nose_bridge'][3]
point_right_nose3 = face_landmarks['nose_tip'][4]

point_right_eye1 = face_landmarks['right_eye'][3]
point_right_eye2 = face_landmarks['right_eye'][0]

x = np.array([])
y = np.array([])
x_right = np.array([])
y_right = np.array([])

point_inter_1_x, point_inter_1_y = line_intersection((point_chin1, point_nose1), (point_eye1, point_lip1))
x = np.append(x, point_inter_1_x)
y = np.append(y, point_inter_1_y)
point_inter_2_x, point_inter_2_y = line_intersection((point_chin1, point_nose2), (point_eye2, point_lip1))
x = np.append(x, point_inter_2_x)
y = np.append(y, point_inter_2_y)
point_inter_3_x, point_inter_3_y = line_intersection((point_eye1, point_lip1), (point_chin2, point_nose3))
x = np.append(x, point_inter_3_x)
y = np.append(y, point_inter_3_y)
point_inter_4_x, point_inter_4_y = line_intersection((point_chin1, point_lip1), (point_chin3, point_eye1))
x = np.append(x, point_inter_4_x)
y = np.append(y, point_inter_4_y)

x = np.append(x, point_inter_1_x)
y = np.append(y, point_inter_1_y)
print(x)
print(y)

point_inter_right_1_x, point_inter_right_1_y = line_intersection((point_right_chin1, point_right_nose1), (point_right_eye1, point_right_lip1))
x_right = np.append(x_right, point_inter_right_1_x)
y_right = np.append(y_right, point_inter_right_1_y)
point_inter_right_2_x, point_inter_right_2_y = line_intersection((point_right_chin1, point_right_nose2), (point_right_eye2, point_right_lip1))
x_right = np.append(x_right, point_inter_right_2_x)
y_right = np.append(y_right, point_inter_right_2_y)
point_inter_right_3_x, point_inter_right_3_y = line_intersection((point_right_eye1, point_right_lip1), (point_right_chin2, point_right_nose3))
x_right = np.append(x_right, point_inter_right_3_x)
y_right = np.append(y_right, point_inter_right_3_y)
point_inter_right_4_x, point_inter_right_4_y = line_intersection((point_right_chin1, point_right_lip1), (point_right_chin3, point_right_eye1))
x_right = np.append(x_right, point_inter_right_4_x)
y_right = np.append(y_right, point_inter_right_4_y)

x_right = np.append(x_right, point_inter_right_1_x)
y_right = np.append(y_right, point_inter_right_1_y)

x_copy = x.copy()
y_copy = y.copy()

x, y = get_boundary_points(x, y)
x, y = get_interior_points(x, y)
print(x)
print(y)
x_len = len(x)


fig = plt.figure()

im = img_rgb.copy()
imOrg = img_rgb.copy()

im = apply_blush_color(im)
imOrg = smoothen_blush(im, imOrg, x, y)
imOrg = smoothen_blush(im, imOrg, x_right, y_right)

fig.add_subplot(1, 2, 1)
plt.imshow(imOrg)

fig.add_subplot(1, 2, 2)
plt.imshow(img_rgb)

cv2.imwrite('output2.jpg', cv2.cvtColor(imOrg, cv2.COLOR_RGB2BGR))


plt.imshow(img_rgb)
plt.show()



