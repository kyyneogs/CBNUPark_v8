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

        model = YOLO(f'weights/{self.model_name}.pt')
        model.fuse()

        return model
    
    def predict(self, frame):

        results = self.model(frame)

        return results
    
    def plot_bboxes(self, results, frame, classes=None):

        # Extract detections
        for result in results:
            boxes = result.boxes.cpu().numpy()

            class_ids = boxes.cls
            confs = boxes.conf
            xyxys = boxes.xyxy

            for idx in range(len(class_ids)):
                if class_ids[idx] in classes:
                    xyxy = xyxys[idx]
                    cv2.rectangle(frame,(int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])),\
                                        (0,255,0), thickness=1, lineType=cv2.LINE_AA)
        
        # None return