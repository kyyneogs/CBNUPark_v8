import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO


class ObjectDetector:

    def __init__(self, model_name):
        # object detector init
        
        self.model_name = model_name

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Selected Device: ', self.device)

        self.model = self.load_model()


    def load_model(self):

        model = YOLO(f'weights/{self.model_name}')

        model.fuse()

        return model
    
    def predict(self, frame, conf: float = 0.15, classes: list = None):
        """
        predict object from frame function.
        return detection results.

        frame : cam frame
        conf : confidence, default = 0.15
        classes : class filter, default = None
        """
        results = self.model(frame, conf=conf, classes=classes)

        return results
    
    def plot_bboxes(self, results, frame, color = (255,0,0)):
        """
        plot bboxes on frame with results.

        results : detection results
        frame : cam frame
        color : color (B,G,R), defaults = BLUE
        """
        tl = 1
        t_size = cv2.getTextSize('vehicle:00.00', 0, fontScale=tl / 3, thickness=tl)[0]

        # Extract detections
        for result in results:

            boxes = result.boxes.cpu().numpy()
            class_ids = boxes.cls
            confs = boxes.conf
            xyxys = boxes.xyxy

            for idx in range(len(class_ids)):

                label = 'vehicle:' + f'{confs[idx]:.2f}'
                xyxy = xyxys[idx]
                pts = [(int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), \
                       (int(xyxy[0])+t_size[0], int(xyxy[1])-t_size[1]-3)]

                cv2.rectangle(frame,pts[0], pts[1], color, thickness=tl, lineType=cv2.LINE_AA)
                cv2.rectangle(frame, pts[0], pts[2], color, -1, cv2.LINE_AA)
                cv2.putText(frame, label, (pts[0][0], pts[0][1]-2), 0, tl/3, \
                            (255,255,255), thickness=tl, lineType=cv2.LINE_AA)