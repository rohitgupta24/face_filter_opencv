import dlib
import cv2

img = cv2.imread('1.jpg')

#shape predictor using save file
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#front face detector
detector = dlib.get_frontal_face_detector()
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(img_grey)

#rectangle on front face
for face in faces:
    x1,y1 = face.left(),face.top()
    x2,y2 = face.right(),face.bottom()

    img = cv2.rectangle(img ,(x1,y1), (x2,y2),(0,255,0),(3))
    landmarks = predictor(img_grey,face)
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img,(x,y),5,(39,78,255),cv2.FILLED)


cv2.imshow("original", img)
cv2.waitKey(0)

