import cv2
from matplotlib import pyplot as plt
import face_recognition
import numpy as np
from scipy.interpolate import interp1d
from numpy.linalg import lstsq

img_bgr = cv2.imread('woman4.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

face_landmarks = face_recognition.face_landmarks(img_rgb)[0]




# i_eye = 3
# for face_part, face_part_coords in face_landmarks.items():
#     if face_part == 'left_eye' or face_part == 'right_eye' or face_part == 'left_eyebrow' or face_part == 'right_eyebrow':
#         if face_part == 'left_eye':
#             for coord in face_part_coords:
#                 cv2.circle(img_rgb, coord, radius = 1, color= (0, 0, 0), thickness = -1)
#                 i_eye = i_eye + 1
#         else:
#             for coord in face_part_coords:
#                 cv2.circle(img_rgb, coord, radius=1, color=(0, 0, 0), thickness=-1)

left_eye_points = face_landmarks['left_eye']
x_interp_left_eye = np.array([])
y_interp_left_eye = np.array([])

x_interp_left_eye = np.append(x_interp_left_eye, left_eye_points[0][0])
y_interp_left_eye = np.append(y_interp_left_eye, left_eye_points[0][1])

x_interp_left_eye = np.append(x_interp_left_eye, left_eye_points[1][0])
y_interp_left_eye = np.append(y_interp_left_eye, left_eye_points[1][1])

x_interp_left_eye = np.append(x_interp_left_eye, left_eye_points[2][0])
y_interp_left_eye = np.append(y_interp_left_eye, left_eye_points[2][1])

x_interp_left_eye = np.append(x_interp_left_eye, left_eye_points[3][0])
y_interp_left_eye = np.append(y_interp_left_eye, left_eye_points[3][1])

f = interp1d(x_interp_left_eye, y_interp_left_eye, kind = 'cubic')

xnew_interp_left_eye = np.linspace(x_interp_left_eye[0], x_interp_left_eye[3], num = 41, endpoint= True)


#for eye_liner
points_poly_eyeliner = []
points_poly_second = []
points_poly_third = []
points_interp_top_x = []
points_interp_top_y = []
points_interp_bot_x = []
points_interp_bot_y = []


point_for_between_eye_and_brow_1 = face_landmarks['left_eyebrow'][0]
point_for_between_eye_and_brow_2 = face_landmarks['left_eye'][0]
points_interp_bot_x.append(face_landmarks['left_eye'][0][0])
points_interp_bot_y.append(face_landmarks['left_eye'][0][1])

points_poly_eyeliner.append(list(point_for_between_eye_and_brow_2))
#points_poly_eyeliner = np.append(points_poly_eyeliner, [point_for_between_eye_and_brow_2[0], point_for_between_eye_and_brow_2[1]])
point_between_eye_and_brow = (int(point_for_between_eye_and_brow_1[0] / 7 + point_for_between_eye_and_brow_2[0] * 6 / 7), int(point_for_between_eye_and_brow_1[1] / 7 + point_for_between_eye_and_brow_2[1] * 6 / 7))
#points_poly_eyeliner = np.append(points_poly_eyeliner, [point_between_eye_and_brow[0], point_between_eye_and_brow[1]])
points_poly_eyeliner.append(list(point_between_eye_and_brow))
vector_difference_between_eye_and_brow = (int((point_for_between_eye_and_brow_1[0] - point_for_between_eye_and_brow_2[0]) / 10), int((point_for_between_eye_and_brow_1[1] - point_for_between_eye_and_brow_2[1]) / 10))

vector_difference_between_eye_and_brow_1 = (int(vector_difference_between_eye_and_brow[0] * 2/3 + face_landmarks['left_eye'][1][0]), int(vector_difference_between_eye_and_brow[1] * 2/3 + face_landmarks['left_eye'][1][1]))
#points_poly_eyeliner = np.append(points_poly_eyeliner, [vector_difference_between_eye_and_brow_1[0], vector_difference_between_eye_and_brow_1[1]])
points_poly_eyeliner.append(list(vector_difference_between_eye_and_brow_1))
points_poly_second.append(list(vector_difference_between_eye_and_brow_1))
#points_poly_eyeliner = np.append(points_poly_eyeliner, [face_landmarks['left_eye'][1][0], face_landmarks['left_eye'][1][1]])
points_poly_eyeliner.append(list(face_landmarks['left_eye'][1]))
points_poly_second.append(list(face_landmarks['left_eye'][1]))

points_interp_bot_x.append(face_landmarks['left_eye'][1][0])
points_interp_bot_y.append(face_landmarks['left_eye'][1][1])

points_poly_first = np.asarray(points_poly_eyeliner)
##cv2.fillPoly(img_rgb, [points_poly_first], (0, 0, 0), 8)


vector_difference_between_eye_and_brow_2 = (int(vector_difference_between_eye_and_brow[0] / 3 + face_landmarks['left_eye'][2][0]), int(vector_difference_between_eye_and_brow[1] / 3 + face_landmarks['left_eye'][1][1]))
points_poly_eyeliner.append(list(face_landmarks['left_eye'][2]))
points_poly_second.append(list(face_landmarks['left_eye'][2]))
points_poly_third.append(list(face_landmarks['left_eye'][2]))
points_interp_bot_x.append(face_landmarks['left_eye'][2][0])
points_interp_bot_y.append(face_landmarks['left_eye'][2][1])

points_poly_eyeliner.append(list(vector_difference_between_eye_and_brow_2))
points_poly_second.append(list(vector_difference_between_eye_and_brow_2))
points_poly_third.append(list(vector_difference_between_eye_and_brow_2))

points_poly_second = np.asarray(points_poly_second)

##cv2.fillPoly(img_rgb, [points_poly_second], (0, 0, 0), 8)

points_poly_eyeliner.append(list(face_landmarks['left_eye'][3]))
points_poly_third.append(list(face_landmarks['left_eye'][3]))

points_interp_bot_x.append(face_landmarks['left_eye'][3][0])
points_interp_bot_y.append(face_landmarks['left_eye'][3][1])

points_poly_third = np.asarray(points_poly_third)
##cv2.fillPoly(img_rgb, [points_poly_third], (0, 0, 0), 8)

points_interp_top_x.append(point_between_eye_and_brow[0])
points_interp_top_y.append(point_between_eye_and_brow[1])

points_interp_top_x.append(vector_difference_between_eye_and_brow_1[0])
points_interp_top_y.append(vector_difference_between_eye_and_brow_1[1])

points_interp_top_x.append(vector_difference_between_eye_and_brow_2[0])
points_interp_top_y.append(vector_difference_between_eye_and_brow_2[1])

points_interp_top_x.append(face_landmarks['left_eye'][3][0])
points_interp_top_y.append(face_landmarks['left_eye'][3][1])

f3 = interp1d(points_interp_top_x, points_interp_top_y, 'cubic')
x_f3_interv = np.linspace(points_interp_top_x[0], points_interp_top_x[3], num=41, endpoint=True)

f_eyeline_bot = interp1d(points_interp_bot_x, points_interp_bot_y, 'cubic')
x_f_eyeline_bot_interv = np.linspace(points_interp_bot_x[0], points_interp_bot_x[3], num=41, endpoint=True)

##print('points eyeliner')
##print(points_poly_eyeliner)



points_poly_eyeliner_numpy = np.asarray(points_poly_eyeliner)

# for point in points_poly_eyeliner_numpy:
#     cv2.circle(img_rgb, tuple(point), radius=4, color=(255, 255, 255), thickness = -1)

points_poly_eyeliner_numpy = points_poly_eyeliner_numpy.reshape((-1, 1, 2))
# cv2.fillPoly(img_rgb, [points_poly_eyeliner_numpy], (0, 0, 0), 8)


# cv2.circle(img_rgb, vector_difference_between_eye_and_brow_1, radius=3, color=(128, 0, 128), thickness = -1)
# cv2.circle(img_rgb, vector_difference_between_eye_and_brow_2, radius=3, color=(128, 0, 128), thickness = -1)

# cv2.circle(img_rgb, point_between_eye_and_brow, radius=1, color=(128, 0, 128), thickness=-1)
#for eye_liner2
point_interp_for_eye_liner_1 = point_between_eye_and_brow
point_interp_for_eye_liner_2 = face_landmarks['left_eye'][3]

interp_array_eye_liner_x = np.array([point_interp_for_eye_liner_1[0], point_interp_for_eye_liner_2[0]])
interp_array_eye_liner_y = np.array([point_interp_for_eye_liner_1[1], point_interp_for_eye_liner_2[1]])


f1 = interp1d(interp_array_eye_liner_x, interp_array_eye_liner_y)
xnew_interp_eye_liner = np.linspace(interp_array_eye_liner_x[0], interp_array_eye_liner_x[1], num = 41, endpoint = True)

xnew_interp_left_eye_len = len(xnew_interp_left_eye)

# for i in range(xnew_interp_left_eye_len):
#     cv2.circle(img_rgb, center= (int(xnew_interp_left_eye[i]), int(f(xnew_interp_left_eye)[i])), radius=1, color=(0, 0, 0), thickness=-1)


##cv2.polylines(img_rgb, points_poly_eyeliner_numpy, True, (0, 0, 0), 3)



#Filling the area between curves
start_interv_right = 0
start_interv_left = 0
end_interv_right = 0
end_interv_left = 0
if x_f3_interv[0] > x_f_eyeline_bot_interv[0]:
    start_interv_right = x_f3_interv[0]
    start_interv_left = x_f_eyeline_bot_interv[0]
    end_interv_left = x_f3_interv[0]
else:
    start_interv_right = x_f_eyeline_bot_interv[0]
    start_interv_left = x_f3_interv[0]
    end_interv_left = x_f_eyeline_bot_interv[0]

if x_f3_interv[len(x_f3_interv) - 1] > x_f_eyeline_bot_interv[len(x_f_eyeline_bot_interv) - 1]:
    end_interv_right = x_f_eyeline_bot_interv[len(x_f_eyeline_bot_interv) - 1]
else:
    end_interv_right = x_f3_interv[len(x_f3_interv) - 1]

x_interv_between_right = np.linspace(start_interv_right, end_interv_right, num=81, endpoint=True)

y_low_left = f_eyeline_bot(x_f_eyeline_bot_interv[0])
y_top_left = f3(x_f3_interv[0])

x_interv_between_left = np.array([start_interv_left, end_interv_left])
y_interv_between_left = np.array([y_top_left, y_low_left])

A_interv_between = np.vstack([x_interv_between_left, np.ones(len(x_interv_between_left))]).T
m, c = lstsq(A_interv_between, y_interv_between_left, rcond=None)[0]

x_interv_between_left = np.linspace(start_interv_left, end_interv_left, num = 41, endpoint=True)
y_interv_between_left = m * x_interv_between_left + c



plt.imshow(img_rgb)
poly_collection = plt.fill_between(x_interv_between_left, y_interv_between_left, f3(x_interv_between_left), color = 'black')
print(poly_collection.get_paths()[0].vertices)
points_path_x = poly_collection.get_paths()[0].vertices[:, 0]
points_path_y = poly_collection.get_paths()[0].vertices[:, 1]
#plt.plot(points_path_x, points_path_y, 'bo')

# plt.plot(x_f3_interv, f3(x_f3_interv), '-', x_f_eyeline_bot_interv, f_eyeline_bot(x_f_eyeline_bot_interv), '-', x_interv_between_left, y_interv_between_left, '--')
poly_collection2 = plt.fill_between(x_interv_between_right, f3(x_interv_between_right), f_eyeline_bot(x_interv_between_right), color='black')
points_path_2_x = poly_collection2.get_paths()[0].vertices[:, 0]
points_path_2_y = poly_collection2.get_paths()[0].vertices[:, 1]
#plt.plot(points_path_2_x, points_path_2_y, 'go')

#print(plt.fill_between(x_interv_between_left, y_interv_between_left, f3(x_interv_between_left), color = 'black'))
#plt.plot(x_interp_left_eye, y_interp_left_eye, 'o', xnew_interp_left_eye, f(xnew_interp_left_eye), '-')
plt.show()

