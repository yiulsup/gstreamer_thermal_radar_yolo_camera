import cv2
import numpy as np 

cap = cv2.VideoCapture("/dev/video2")

while True:
    ret, frame = cap.read()
    cv2.imshow('d', frame)
    cv2.waitKey(1)