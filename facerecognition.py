import face_recognition
import cv2
import time
from os import listdir
import numpy as np
import datetime
import os
import sys
import pickle
face_locations = []
face_encodings = []
face_names = []

data = pickle.loads(open("encodings.pickle","rb").read())
process_this_frame = True
video_capture = cv2.VideoCapture(0)
while True:
    # Grab a single frame of video
    #print('Capturing image')
    ret, frame = video_capture.read()
    # Resize and gray scale frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    image_gray = cv2.cvtColor(small_frame,cv2.COLOR_BGR2GRAY)

    if(process_this_frame):
        face_locations = face_recognition.face_locations(image_gray) #get location by hog using grayscale image
        face_encodings = face_recognition.face_encodings(small_frame, face_locations) #get encodings by rgb images
        if(face_encodings):
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)

                matches = face_recognition.compare_faces(data["encodings"],face_encoding,0.6)
                name = "Unknown"
                #check to see if we have found a match
                if True in matches:
                    matchIdx = [i for(i,b) in enumerate(matches) if b]#only get the True index of matches
                    counts = {}
                    for i in matchIdx:
                        name = data["names"][i] #get the name at index i
                        counts[name] = counts.get(name,0) + 1#put it into a dictionary
                    #get the fist max index, there would be some wrong cases
                    name = max(counts,key =counts.get) # get the name with max idx

                face_names.append(name) #get the face_names
    process_this_frame = not process_this_frame
      
    # Display the results with the condition of 1 face
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)    


    # Display the resulting image

    cv2.imshow('Video', frame) 


    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'): #truely turn off the system
        break

cv2.destroyAllWindows()
video_capture.release()



