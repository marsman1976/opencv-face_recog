# Denne klassen skal inneholde alle funkjonaliteten som skal prosessere innhentet data i den filen fcrecog-sss.py
#Denne klassen skal brukes i den andre fila via å skape et objekt basert på OOP metode


import face_recognition
import cv2
import os
import glob
import numpy as np

class AbdulFace:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Endring av størrelse for mer effektiviteten
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        
        # Importering bildene-Downloading images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Lagring bidene som har blitt kodet-Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Få den filnavn fra fil-sted-bath.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Få den enkoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Lagre den filvavn med enkoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            

           
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
