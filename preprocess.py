import face_recognition
import pickle
import os
import cv2
from os import listdir
known_encodings = []
known_names = []
path = "dataset"
count = 0
# loop over the image paths
for folder in listdir(path):
	image_folder = os.path.join(path,folder)
	count +=1
	for file in listdir(image_folder):
		full_file_path = os.path.join(image_folder,file)
		image = cv2.imread(full_file_path)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale

		boxes = face_recognition.face_locations(gray)

		# compute the facial embedding for the face
		encodings = face_recognition.face_encodings(image, boxes)

		# loop over the encodings
		for encoding in encodings:
			known_encodings.append(encoding)
			known_names.append(folder)

#create a data to store encodings and names
data = {"encodings": known_encodings, "names": known_names}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
print("{} faces trained. Thank you for waiting".format(count))