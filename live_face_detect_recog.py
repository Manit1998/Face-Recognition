import cv2
import face_recognition

cv2.namedWindow("My_image")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, img = vc.read()
else:
    rval = False
manit_image=face_recognition.load_image_file("manit.jpg")
manit_encoding=face_recognition.face_encodings(manit_image)[0]
shiv_image=face_recognition.load_image_file("shiv.jpg")
shiv_encoding=face_recognition.face_encodings(shiv_image)[0]
ankur_image=face_recognition.load_image_file("ankur.jpg")
ankur_encoding=face_recognition.face_encodings(ankur_image)[0]

know_encoding=[manit_encoding,shiv_encoding,ankur_encoding]

know_name=["Manit","Shiv","Ankur"]

face_locations= []
unknown_image_encoding= []



while rval:
    rval, img = vc.read()
    small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    unknown_image_encoding = face_recognition.face_encodings(rgb_small_frame,face_locations)
    name_face=[]
    
    for hel in unknown_image_encoding:
        result = face_recognition.compare_faces(know_encoding, hel)
        name="unknown"
        if True in result:
            i = result.index(True)
            name = know_name[i]

        name_face.append(name)

    for (top, right, bottom, left), name in zip(face_locations, name_face):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.rectangle(img, (left, bottom - 35), (right, bottom), (255, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


        




    cv2.imshow("My_image", img)
    key = cv2.waitKey(20)

    if key == 27:
    	cv2.imwrite("Image.png",img) 
    	break

cv2.destroyWindow("My_image")