# main file emotion detector
from time import sleep
import cv2
import numpy as np
from keras.preprocessing import image
import tensorflow as tf

# add classifier and model path
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = tf.keras.models.load_model('Emotion_detection.h5')

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,6)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)
        ROI_gray = gray[y:y+h, x:x+w]
        ROI_gray = cv2.resize(ROI_gray, (48,48), interpolation=cv2.INTER_AREA)
        
        if np.sum([ROI_gray])!=0:
            ROI = ROI_gray.astype('float')/255
            ROI = image.img_to_array(ROI)
            ROI = np.expand_dims(ROI, axis=0)
            
            preds = classifier.predict(ROI)[0]
            label = class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
        else:
            cv2.putText(gray, 'NO FACE DETECTED', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # to quit press 'q'
        break
        
cap.release()
cv2.destroyAllWindows()
            