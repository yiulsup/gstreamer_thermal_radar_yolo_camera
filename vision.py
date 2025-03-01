import cv2
import numpy as np 

cap = cv2.VideoCapture("/dev/video0")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow('d', frame)
    cv2.waitKey(1)