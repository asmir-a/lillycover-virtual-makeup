from PIL import Image, ImageDraw
import face_recognition

image = face_recognition.load_image_file('woman1.jpg')

face_landmarks_list = face_recognition.face_landmarks(image)

print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)

for facial_feature in face_landmarks_list[0].keys():
    print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks_list[0][facial_feature]))

for facial_feature in face_landmarks_list[0].keys():
    d.line(face_landmarks_list[0][facial_feature], width=1)

pil_image.show()