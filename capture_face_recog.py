import cv2
import face_recognition
cv2.namedWindow("Press C for Capture Image")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False


while rval:
    cv2.imshow("Press C for Capture Image",frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    face=cv2.CascadeClassifier("/usr/local/lib/python3.5/dist-packages/cv2/data/haarcascade_frontalface_default.xml")
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    faces=face.detectMultiScale(grey,1.3,9)
    for (x,y,w,h) in faces:
    	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    if key == ord("c") or key==27 :
    	cv2.imwrite("Image.png",frame) 
    	break
cv2.destroyWindow("Press C for Capture Image")
print("Please Wait for image recognition........")

manit_image=face_recognition.load_image_file("manit.jpg")
manit_encoding=face_recognition.face_encodings(manit_image)[0]
shiv_image=face_recognition.load_image_file("shiv.jpg")
shiv_encoding=face_recognition.face_encodings(shiv_image)[0]
ankur_image=face_recognition.load_image_file("ankur.jpg")
ankur_encoding=face_recognition.face_encodings(ankur_image)[0]


know_encoding=[manit_encoding,shiv_encoding,ankur_encoding]

know_name=["Manit","Shiv","Ankur"]
unknown_image=face_recognition.load_image_file("Image.png")
unknown_encoding=face_recognition.face_encodings(unknown_image)[0]
result=face_recognition.compare_faces(know_encoding,unknown_encoding)
try :
    if True in result:
        i=result.index(True)
        print("It is a picture of :",know_name[i])
except IndexError:
    print("Unknown Picture ")

