
import os
import cv2
import numpy as np
import pyttsx3
import face_train as ft

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait() 

face_classifier = cv2.CascadeClassifier('C:/Users/Madhuri/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def face_extractor(img) :
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,6)
    
    if faces is():
        return None
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h,x:x+w]
        
    return cropped_face
    

cap = cv2.VideoCapture(0)
count = 0

root_path = "E:/B.Tech Study/TY IT/FaceDetection/data/"

speak("please tell your department")
dept_name = input("department : ")

speak("please tell your name!")
student_name = input("name : ")

#new_folder = "user"+str(np.random.randint(100))
dept_path = os.path.join(root_path, dept_name)
try:
    os.mkdir(dept_path)
except Exception as e:
    pass
    
name_path = os.path.join(dept_path, student_name)
os.mkdir(name_path)

images_path = os.path.join(name_path, 'images')
os.mkdir(images_path)

speak("collecting samples,..! please wait!..")
speak("please! look at the camera only")   

while True :
    ret,frame = cap.read()

    if face_extractor(frame) is not None :
        count += 1
        
        face = cv2.resize(face_extractor(frame), (200,200))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        
        i_path = images_path + "/img_"+str(count)+".jpg"
        
        cv2.imwrite (i_path,face)
        
        #cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow ('face cropper',face)
        
    else:
        print('face not found')
        pass
    
    if cv2.waitKey(1) == 13 or count == 100:
        break
        
cap.release()
cv2.destroyAllWindows()
print('collecting samples completed')
speak("all samples are collected!.. thank you!..")

#modified from here
model = ft.faceTrain(dept_name,student_name)

model.save('E:/B.Tech Study/TY IT/FaceDetection/data/'+dept_name+'/'+student_name+'/trained_model.xml')

