import cv2
from mtcnn import MTCNN
import os
import numpy as numpy

detector = MTCNN()

path = '/media/ml/Data Disk/upscaling/celebA_data/augmented/'
pathout = '/media/ml/Data Disk/upscaling/celebA_data/faces/'
classes = os.listdir(path)

for person in classes:

    images = os.listdir(path + person)
    c = 0
    os.mkdir(pathout + person)
    print(person)
    for image in images:

        frame = cv2.imread(path + person + '/' + image)
        #cv2.imshow('frame', frame)
        detect = detector.detect_faces(frame)

        if detect:
            boxes = detect[0]['box']
            x,y,w,h = boxes
            extract = frame[y:y+h, x:x+w]
            cv2.imwrite(pathout + person + '/' + str(c) + '.jpg', extract)
            c += 1
            #cv2.imshow('frame', extract)
            #cv2.waitKey()
            #print('yes', boxes)

