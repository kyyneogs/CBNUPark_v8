import torch
import numpy as np
import cv2
from ultralytics import YOLO
from detector import ObjectDetector

# coco datasets classes: [ 3:car, 4:motorcycle, 6:bus, 8:truck ]
CLASSES = [2, 3, 5, 7]
MODEL = 'yolov8m'

# load yolov8 model
# model = YOLO(f'weights/{MODEL}.pt')

# load detector
detector = ObjectDetector(MODEL)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    assert ret

    # predict object
    results = detector.predict(frame)
    detector.plot_bboxes(results=results, frame=frame, classes=CLASSES)
    # track object
    # results = model.predict(frame, imgsz=640, conf=0.25, classes=CLASSES)
    # post_frame = results[0].plot()

    cv2.imshow('frame', frame)
    if cv2.waitKey(30)==27:
        break

cap.release()
cv2.destroyAllWindows()