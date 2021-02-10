from PIL import Image, ImageDraw
import face_recognition
import cv2
from matplotlib import pyplot as plt



# Create a PIL imagedraw object so we can draw on the picture
img = cv2.imread('woman1.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

face_landmarks = (face_recognition.face_landmarks(img_rgb))[0]
print(face_landmarks)
print(face_landmarks['chin'][14][0])

i = 0
for feature in face_landmarks.keys():
    for feature_point in face_landmarks[feature]:
        if feature == 'nose_tip':
            cv2.circle(img_rgb, feature_point, radius = int(i * 0.5), color=(255,0,0), thickness = 1)
            i = i + 1
        else:
            cv2.circle(img_rgb, feature_point, radius = 3, color=(0, 0, 255), thickness = -1)


# cv2.circle(img_rgb, face_landmarks['top_lip'][6], radius= 3, color=(255, 255, 255), thickness=1)
#
# blush_point_1_x = int((face_landmarks['chin'][14][0]+face_landmarks['top_lip'][6][0]) / 2)
# blush_point_1_y = int((face_landmarks['chin'][14][1] + face_landmarks['top_lip'][6][1]) / 2)
# cv2.circle(img_rgb, (blush_point_1_x, blush_point_1_y), radius=0, color = (255, 255, 255), thickness = -1)
#
# blush_point_2_x = int((face_landmarks['chin'][14][0]+face_landmarks['right_eye'][3][0]) / 2)
# blush_point_2_y = int((face_landmarks['chin'][14][1] + face_landmarks['right_eye'][3][1]) / 2)
# cv2.circle(img_rgb, (blush_point_2_x, blush_point_2_y), radius=0, color=(255, 255, 255), thickness = -1)
#
# blush_point_3_x = int((face_landmarks['chin'][14][0] + face_landmarks['nose_tip'][4][0]) / 2)
# blush_point_3_y = int((face_landmarks['chin'][14][1] + face_landmarks['nose_tip'][4][1]) / 2)
# cv2.circle(img_rgb, (blush_point_3_x, blush_point_3_y), radius=0, color=(255, 255, 255), thickness = -1)

plt.imshow(img_rgb)
plt.show()