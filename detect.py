import cv2
import cv2 as cv
import numpy as np
import urllib.request

x = int(input('Where do you get the input? \n 0 = local \n 1 = url \n 2 = cam \n Answer(Number) = '))

face_model = cv.CascadeClassifier('facedetectionmodel.xml')

def detection():
    gray_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_model.detectMultiScale(gray_scale)

    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2) #!RGB --> BGR


    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def detection2(img):
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_model.detectMultiScale(gray_scale, 1.3, 5)
    if faces is ():
        return img
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2) #!RGB --> BGR
    return img


if x == 0:
    img = cv.imread('tag.ong')
    detection()

elif x == 1:
    img_url = str(input('url = '))
    #img_url = input('url photo = ')
    # ดาวน์โหลดภาพจาก URL
    with urllib.request.urlopen(img_url) as url_response:
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        # แปลงข้อมูลภาพเป็น array
        img = cv.imdecode(img_array, -1)
    detection()

elif x == 2:
    cam_select = int(input('Which camera? \n (if you have only 1 cam. enter 0, Default = 0) \n Answer = '))
    img = cv2.VideoCapture(cam_select)
    #cam = cv2.VideoCapture(0)

    while True:
        ret, frame = img.read()
        frame = detection2(frame)

        cv2.imshow('Face Detection Cam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): #? กด Q เพื่อออก
            break

    img.release()
    cv2.destroyAllWindows()
