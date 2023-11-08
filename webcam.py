import torch
import numpy as np
import cv2
from ultralytics import YOLO
from detector import ObjectDetector

# coco datasets classes: [ 3:car, 4:motorcycle, 6:bus, 8:truck ]
CLASSES = [2, 3, 5, 7]
MODEL = 'yolov8m.pt'
COLOR = (255, 0, 0)

# load detector with yolov8 model
detector = ObjectDetector(MODEL)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    
    ret, frame = cap.read()
    assert ret

    # predict object
    results = detector.predict(frame, classes=CLASSES)
    detector.plot_bboxes(results=results, frame=frame, color=COLOR)
    # track object
    # results = model.predict(frame, imgsz=640, conf=0.25, classes=CLASSES)
    # post_frame = results[0].plot()

    cv2.imshow('frame', frame)
    if cv2.waitKey(30)==27:
        break

cap.release()
cv2.destroyAllWindows()