import dlib
import cv2
import numpy as np

img = cv2.imread('1.jpg')


def createBox(img, points, scale=5, show=False):
    bbox = cv2.boundingRect(points)
    x, y, w, h = bbox
    imgCrop = img[y:y + h, x:x + w]
    imgCrop = cv2.resize(imgCrop, (0, 0), None, scale, scale)
    return imgCrop

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

    myPoints =[]

    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y

        myPoints.append([x, y])
        
        cv2.circle(img,(x,y),5,(39,78,255),cv2.FILLED)
        # cv2.putText(img,str(n),(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,0,255),1) 

    myPoints = np.array(myPoints)
    imgLeftEye = createBox(img, myPoints[36:42])
    imgLips = createBox(img, myPoints[48:61])
    cv2.imshow('Lips', imgLips)
    cv2.imshow('Left Eye', imgLeftEye)

cv2.imshow("original", img)
cv2.waitKey(0)

