import cv2
import dlib
import numpy as np

image_path = '/home/cuab/Pictures/Screenshots/150.jpg'
img = cv2.imread(image_path)
if img.any():
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # Face detector + 81-point shape predictor
    detector = dlib.get_frontal_face_detector()
    dlib_dat_path = '/home/cuab/Documents/shape_predictor_81_face_landmarks/shape_predictor_81_face_landmarks.dat'
    predictor = dlib.shape_predictor(dlib_dat_path)

# # Detect faces
# faces = detector(gray)

# face_dict = {
#     'forehead': [68, 69, 70, 71, 72, 73, 74, 75, 76, 77],
#     "left_eye": [36, 37, 38, 39],
#     'right_eye': [42, 43, 44, 45],
#     'nose': [27, 28, 29, 30, 31, 32, 33, 34, 35],
#     'mouth': [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67],
#     'chin': [4, 5, 6, 7, 8, 9, 10, 11, 12]
# }

# for face in faces:
#     landmarks = predictor(gray, face)
#     print(landmarks.num_parts)  # should print 81

#     for i in range(81):
#         if i in face_dict['chin']:
#             x = landmarks.part(i).x
#             y = landmarks.part(i).y
#             cv2.circle(img, (x, y), 2, (234, 255, 0), -1)

# cv2.imshow("81 Landmarks", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

    faces = detector(gray_img)
    print(faces)

    if len(faces) == 0:
        print('No face detected')
    else:
        landmarks = predictor(gray_img, faces[0])
        points = np.array([[f.x, f.y] for f in landmarks.parts()])
        print(len(points))
else:
    print('No image imported')