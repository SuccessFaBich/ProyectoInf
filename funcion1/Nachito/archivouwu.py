import face_recogntion as fr
import os
import cv2 as cv
from time import sleep

def get_encoded_faces():
    """
    looks trhough the faces folder and encodes all the faces

    :return: dict of (name, image encoded)
    """
    encoded={}

    for dirpath, dnames, fnames in os .walk("./faces"):
        for f in fnames:
            if f .endswith(".jpg") or f .endswith(".png"):
                face= fr .load_image_file("faces/"+f)
                encoding= fr .face_encodings (face) [0]
                encoded[f.split(".")[0]]=encoding
return encoded
