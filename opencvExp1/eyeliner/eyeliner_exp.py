import numpy as np
import cv2
from matplotlib import pyplot as plt
import face_recognition
from scipy.interpolate import interp1d
from numpy.linalg import lstsq
import itertools


def find_points_eyeliner_left(points_eyebrow, points_eye):
    x_eyebrow_0 = points_eyebrow[0][0]
    y_eyebrow_0 = points_eyebrow[0][1]
    x_eye_0 = points_eye[0][0]
    y_eye_0 = points_eye[0][1]
    x_eyebrow_1 = points_eyebrow[1][0]
    y_eyebrow_1 = points_eyebrow[1][1]
    x_eye_1 = points_eye[1][0]
    y_eye_1 = points_eye[1][1]
    x_eyebrow_2 = points_eyebrow[2][0]
    y_eyebrow_2 = points_eyebrow[2][1]
    x_eye_2 = points_eye[2][0]
    y_eye_2 = points_eye[2][1]
    x_eyebrow_3 = points_eyebrow[3][0]
    y_eyebrow_3 = points_eyebrow[3][1]
    x_eye_3 = points_eye[3][0]
    y_eye_3 = points_eye[3][1]

    point_between_brow_and_eye_0 = (int(x_eyebrow_0 / 7 + x_eye_0 * 6 / 7), int(y_eyebrow_0 / 7 + y_eye_0 * 6 / 7))
    vector_difference = (int((x_eyebrow_0 - x_eye_0) / 10), int((y_eyebrow_0 - y_eye_0) / 10))
    vector_difference_x = vector_difference[0]
    vector_difference_y = vector_difference[1]

    point_between_brow_and_eye_1 = (
    int(vector_difference_x * 2 / 3 + x_eye_1), int(vector_difference_y * 2 / 3 + y_eye_1))
    point_between_brow_and_eye_2 = (int(vector_difference_x / 3 + x_eye_2), int(vector_difference_y / 3 + y_eye_2))
    point_between_brow_and_eye_3 = (int(x_eye_3), int(y_eye_3))

    points_poly_eyeliner_eye = np.array([(x_eye_0, y_eye_0), (x_eye_1, y_eye_1), (x_eye_2, y_eye_2), (x_eye_3, y_eye_3)])
    points_poly_eyeliner_eyebrow = np.array([point_between_brow_and_eye_0, point_between_brow_and_eye_1, point_between_brow_and_eye_2,point_between_brow_and_eye_3])
    return points_poly_eyeliner_eye, points_poly_eyeliner_eyebrow
def find_points_eyeliner_right(points_eyebrow, points_eye):
    x_eyebrow0 = points_eyebrow[4][0]
    y_eyebrow0 = points_eyebrow[4][1]
    x_eye0 = points_eye[3][0]
    y_eye0 = points_eye[3][1]
    x_eyebrow1 = points_eyebrow[3][0]
    y_eyebrow1 = points_eyebrow[3][1]
    x_eye1 = points_eye[2][0]
    y_eye1 = points_eye[2][1]
    x_eyebrow2 = points_eyebrow[2][0]
    y_eyebrow2 = points_eyebrow[2][1]
    x_eye2 = points_eye[1][0]
    y_eye2 = points_eye[1][1]
    x_eyebrow3 = points_eyebrow[1][0]
    y_eyebrow3 = points_eyebrow[1][1]
    x_eye3 = points_eye[0][0]
    y_eye3 = points_eye[0][1]

    point_between_brow_and_eye_0 = (int(x_eyebrow0 / 7 + x_eye0 * 6 / 7), int(y_eyebrow0 / 7 + y_eye0 * 6 / 7))
    vector_difference = (int((x_eyebrow0 - x_eye0) / 10), int((y_eyebrow0 - y_eye0) / 10))
    vector_difference_x = vector_difference[0]
    vector_difference_y = vector_difference[1]

    point_between_brow_and_eye_1 = (
        int(vector_difference_x * 2 / 3 + x_eye1), int(vector_difference_y * 2 / 3 + y_eye1))
    point_between_brow_and_eye_2 = (int(vector_difference_x / 3 + x_eye2), int(vector_difference_y / 3 + y_eye2))
    point_between_brow_and_eye_3 = (int(x_eye3), int(y_eye3))

    points_poly_eyeliner_eye = np.array([(x_eye0, y_eye0), (x_eye1, y_eye1), (x_eye2, y_eye2), (x_eye3, y_eye3)])
    points_poly_eyeliner_eyebrow = np.array(
        [point_between_brow_and_eye_0, point_between_brow_and_eye_1, point_between_brow_and_eye_2,
         point_between_brow_and_eye_3])
    return points_poly_eyeliner_eye, points_poly_eyeliner_eyebrow

def draw_points_eyeliner(img, points_eyeliner):
    img_copy = img.copy()
    for x, y in points_eyeliner:
        cv2.circle(img_copy, (x, y), radius=1, color=(0, 0, 0), thickness=-1)
    return img_copy

def find_curve_eyeliner(points_eyeliner):
    x_points_eyeliner = points_eyeliner[:, 0]
    y_points_eyeliner = points_eyeliner[:, 1]
    x_min = np.amin(x_points_eyeliner)
    x_max = np.amax(x_points_eyeliner)
    f_top_curve = interp1d(x_points_eyeliner, y_points_eyeliner, 'cubic')
    x_interv_top_curve = np.linspace(x_min, x_max, num=121, endpoint=True)
    return x_interv_top_curve, f_top_curve

def draw_curve(img, x_interv, f):
    img_copy = img.copy()
    y_points = f(x_interv)
    length_x_interv = len(x_interv)
    for i in range(length_x_interv):
        cv2.circle(img_copy, (int(x_interv[i]), int(y_points[i])), radius=1, color=(128, 0, 128), thickness=-1)
    return img_copy

def shape_left_eye_outer(points_above_eye, points_eye, f_above, f_eye):
    point_eye_zero_x = points_eye[0][0]
    point_above_eye_zero_x = points_above_eye[0][0]
    interv_x_left = np.array([])
    y_values_left_curve = np.array([])
    y_values_left_line = np.array([])


    interv_x_left = np.linspace(point_above_eye_zero_x, point_eye_zero_x, num=41, endpoint=True)
    y_values_left_curve = f_above(interv_x_left)
    x_interv_for_ls = np.array([point_above_eye_zero_x, point_eye_zero_x])
    y_interv_for_ls = np.array([f_above(point_above_eye_zero_x), f_eye(point_eye_zero_x)])
    A_for_ls = np.vstack([x_interv_for_ls, np.ones(len(x_interv_for_ls))]).T
    m, c = lstsq(A_for_ls, y_interv_for_ls, rcond=None)[0]
    y_values_left_line = m * interv_x_left + c
    # TODO: Two complete the else statement

    return interv_x_left, y_values_left_line, y_values_left_curve
def shape_left_eye_inner(points_above_eye, points_eye, f_above, f_eye):
    point_eye_zero_x = points_eye[0][0]
    point_above_eye_zero_x = points_above_eye[0][0]
    point_inner_most_x = points_eye[3][0]
    interv_x_right = np.array([])
    y_values_top_curve = np.array([])
    y_values_bot_curve = np.array([])

    interv_x_right = np.linspace(point_eye_zero_x, point_inner_most_x, num=81, endpoint=True)
    y_values_top_curve = f_above(interv_x_right)
    y_values_bot_curve = f_eye(interv_x_right)

    return interv_x_right, y_values_top_curve, y_values_bot_curve

def get_inner_and_outer_points(points_eyebrow, points_eye):

    points_eyeliner_eye, points_eyeliner_eyebrow = find_points_eyeliner_right(points_eyebrow, points_eye)
    x_interv_top_curve_eyeliner, f_top_curve_eyeliner = find_curve_eyeliner(points_eyeliner_eyebrow)
    x_interv_bot_curve_eyeliner, f_bot_curve_eyeliner = find_curve_eyeliner(points_eyeliner_eye)
    x_interv_eye_outer, y_line_eye_outer, y_curve_eye_outer = shape_left_eye_outer(points_eyeliner_eyebrow, points_eyeliner_eye, f_top_curve_eyeliner,f_bot_curve_eyeliner)
    x_interv_eye_inner, y_top_curve_inner, y_bot_curve_inner = shape_left_eye_inner(points_eyeliner_eyebrow, points_eyeliner_eye, f_top_curve_eyeliner, f_bot_curve_eyeliner)

    points_poly_right_eye_fill_outer = []
    points_poly_right_eye_fill_inner = []
    for i in range(len(x_interv_eye_inner)):
        points_poly_right_eye_fill_inner.append([int(x_interv_eye_inner[i]), int(y_top_curve_inner[i])])
    for i in range(len(x_interv_eye_inner)):
        points_poly_right_eye_fill_inner.append([int(x_interv_eye_inner[i]), int(y_bot_curve_inner[i])])
    for i in range(len(x_interv_eye_outer)):
        points_poly_right_eye_fill_outer.append([int(x_interv_eye_outer[i]), int(y_line_eye_outer[i])])
    for i in range(len(x_interv_eye_outer)):
        points_poly_right_eye_fill_outer.append([int(x_interv_eye_outer[i]), int(y_curve_eye_outer[i])])
    points_poly_right_eye_fill_outer.sort()
    points_poly_right_eye_fill_outer = list(k for k, _ in itertools.groupby(points_poly_right_eye_fill_outer))
    points_poly_right_eye_fill_inner.sort()
    points_poly_right_eye_fill_inner = list(k for k, _ in itertools.groupby(points_poly_right_eye_fill_inner))

    return points_poly_right_eye_fill_outer, points_poly_right_eye_fill_inner






img_bgr = cv2.imread('output3.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

face_landmarks = face_recognition.face_landmarks(img_rgb)[0]
points_left_eyebrow = face_landmarks['left_eyebrow']
points_left_eye = face_landmarks['left_eye']
points_right_eyebrow = face_landmarks['right_eyebrow']
points_right_eye = face_landmarks['right_eye']



points_poly_eye_fill_outer, points_poly_eye_fill_inner =  get_inner_and_outer_points(points_right_eyebrow, points_right_eye)
points_poly_eye_fill_inner_len_third = int(len(points_poly_eye_fill_inner) / 3)
chunks = [points_poly_eye_fill_inner[x:x + points_poly_eye_fill_inner_len_third] for x in range(0, len(points_poly_eye_fill_inner), points_poly_eye_fill_inner_len_third)]
points_poly_right_eye_fill_inner0 = chunks[0]
points_poly_right_eye_fill_inner1 = chunks[1]
points_poly_right_eye_fill_inner2 = chunks[2]

points_poly_right_eye_fill_outer = np.array(points_poly_eye_fill_outer)
points_poly_right_eye_fill_inner0 = np.array(points_poly_right_eye_fill_inner0)
points_poly_right_eye_fill_inner1 = np.array(points_poly_right_eye_fill_inner1)
points_poly_right_eye_fill_inner2 = np.array(points_poly_right_eye_fill_inner2)

poly_mask_1 = np.zeros((img_rgb.shape[:2]), dtype=np.uint8)
cv2.fillPoly(poly_mask_1, np.int32([points_poly_right_eye_fill_outer]), (255, 255, 255), 1)
cv2.fillConvexPoly(poly_mask_1, np.int32([points_poly_right_eye_fill_inner0]), (255, 255, 255), 1)
cv2.fillConvexPoly(poly_mask_1, np.int32([points_poly_right_eye_fill_inner1]), (255, 255, 255), 1)
cv2.fillConvexPoly(poly_mask_1, np.int32([points_poly_right_eye_fill_inner2]), (255, 255, 255), 1)

term_add_weighted0 = cv2.bitwise_and(img_rgb, img_rgb, mask=poly_mask_1)

poly_mask2 = np.zeros((img_rgb.shape))
cv2.fillPoly(poly_mask2, np.int32([points_poly_right_eye_fill_outer]), (1, 1, 1), 1)
cv2.fillConvexPoly(poly_mask2, np.int32([points_poly_right_eye_fill_inner0]), (1, 1, 1), 1)
cv2.fillConvexPoly(poly_mask2, np.int32([points_poly_right_eye_fill_inner1]), (1, 1, 1), 1)
cv2.fillConvexPoly(poly_mask2, np.int32([points_poly_right_eye_fill_inner2]), (1, 1, 1), 1)

term_add_weighted0 = cv2.bitwise_and(img_rgb, img_rgb, mask=poly_mask_1)

poly_mask2 = np.zeros((img_rgb.shape))
cv2.fillPoly(poly_mask2, np.int32([points_poly_right_eye_fill_outer]), (1, 1, 1), 1)
cv2.fillConvexPoly(poly_mask2, np.int32([points_poly_right_eye_fill_inner0]), (1, 1, 1), 1)
cv2.fillConvexPoly(poly_mask2, np.int32([points_poly_right_eye_fill_inner1]), (1, 1, 1), 1)
cv2.fillConvexPoly(poly_mask2, np.int32([points_poly_right_eye_fill_inner2]), (1, 1, 1), 1)

term_add_weighted1 = poly_mask2

res0 = cv2.addWeighted(term_add_weighted0, 0.0, term_add_weighted1, 1.0, 0, dtype = 8)
res1 = cv2.addWeighted(term_add_weighted0, 0.2, term_add_weighted1, 0.8, 0, dtype = 8)
res2 = cv2.addWeighted(term_add_weighted0, 0.4, term_add_weighted1, 0.6, 0, dtype = 8)
res3 = cv2.addWeighted(term_add_weighted0, 0.6, term_add_weighted1, 0.4, 0, dtype = 8)
res4 = cv2.addWeighted(term_add_weighted0, 0.8, term_add_weighted1, 0.2, 0, dtype = 8)
res5 = cv2.addWeighted(term_add_weighted0, 1.0, term_add_weighted1, 0.0, 0, dtype = 8)

term_add0 = cv2.addWeighted(term_add_weighted0, 0.3, term_add_weighted1, 0.7, 0, dtype = 8)

term_add1 = img_rgb.copy()
cv2.fillPoly(term_add1, np.int32([points_poly_right_eye_fill_outer]), (0, 0, 0), 1)
cv2.fillConvexPoly(term_add1, np.int32([points_poly_right_eye_fill_inner0]), (0, 0, 0), 1)
cv2.fillConvexPoly(term_add1, np.int32([points_poly_right_eye_fill_inner1]), (0, 0, 0), 1)
cv2.fillConvexPoly(term_add1, np.int32([points_poly_right_eye_fill_inner2]), (0, 0, 0), 1)

res_addition = cv2.add(term_add0, term_add1)

cv2.imwrite('output4.jpg', cv2.cvtColor(res_addition, cv2.COLOR_BGR2RGB))

plt.imshow(res_addition)
plt.show()


