def face2(dept,student):
    import cv2
    import numpy as np
    from os import listdir
    from os.path import isfile, join
    import time
    import pyttsx3
    import face_train as ft
    
    
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    

    def speak(audio):
        engine.say(audio)
        engine.runAndWait()

    #model = ft.faceTrain(dept,student)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read('E:/B.Tech Study/TY IT/FaceDetection/data/'+dept+'/'+student+'/trained_model.xml')


    face_classifier = cv2.CascadeClassifier('C:/Users/Madhuri/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    def face_detector (img,size =.5) :
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale (gray, 1.3, 6)
        
        if faces is ():
            return img,[]
        
        
        for(x,y,w,h) in faces :
            cv2.rectangle(img, (x,y), (x+w,y+h), (100,255,255), 2)
            
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi,(200,200))
        
        return img,roi

    cap = cv2.VideoCapture(0) 

    measure1 = time.time()
    measure2 = time.time()

    count = 1

    while count < 6:
        if measure2 - measure1 >= 1:
            #print("two seconds")
            measure1 = measure2
            measure2 = time.time()
            count += 1
            if count == 6 :
                cap.release()
                cv2.destroyAllWindows()
                speak(" sorry! unable to identify! please try again!")
                break
                
        else:
            measure2 = time.time()
        
                
        ret, frame = cap.read()

        image,face = face_detector(frame)

        try :
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict (face)

            if result[1] < 500:
                confidence =int(100 * (1-(result[1]/300)))
                display_string = str(confidence) + '% confident that it is user'

            cv2.putText(image, display_string, (30,30), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)

            if confidence > 84:
                #cv2.putText(image, 'USER IDENTIFIED', (250, 470), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.imshow('face cropper',image)
                cap.release()
                cv2.destroyAllWindows()
                speak("user identified")
                return True
                break

            else :
                #cv2.putText(image, 'LOCKED', (170,450), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.imshow('face cropper',image)



        except :

            #cv2.putText(image, 'FACE NOT FOUND', (170,450), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            cv2.imshow('face cropper',image)

            pass
        if cv2.waitKey(1) == 13:
            cap.release()
            cv2.destroyAllWindows()
            break
