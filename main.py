import face_recognition as fr
import os
import cv2
import numpy as np
from time import sleep

def get_encoded_faces():
    """
    busca en la carpeta de rostros y codifica todos los rostros

    :return: dict de (nombre, imagen codificada)
    """

    encoded = {} # almacenara los nombres y codificaciones de las caras

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("imgs/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding
    
    return encoded

def classify_face(im):
    """
    will find all of the faces in a given image and label
    them if it knows what they are
    :param im: str of file path
    :return: list of faces names
    """
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
    # img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
    # img = img[:,:,::-1]

    face_locations = fr.face_locations(img)
    unknown_face_encodings = fr.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # see if the face is a match for the known face(s)
        matches = fr.compare_faces(faces_encoded, face_encoding)
        name  = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = fr.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # draw a label with a name below the faces
            cv2.rectangle(img, (left-20, bottom-15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left-20, bottom+15), font, 1.0, (255, 255, 255), 2)

    # display the resulting image
    while True:

        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            return face_names
        
print(classify_face("mari.jpg"))
print("Fin del programa")
