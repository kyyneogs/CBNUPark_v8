import torch
import numpy as np
import cv2
from ultralytics import YOLO

# load yolov8 model
model = YOLO('yolov8m.pt')

ret, cap = True, cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while ret:
    ret, frame = cap.read()

    # track object
    results = model.track(frame, persist=True)
    post_frame = results[0].plot()

    cv2.imshow('frame', post_frame)
    if cv2.waitKey(30)==27:
        break

cap.release()
cv2.destroyAllWindows()