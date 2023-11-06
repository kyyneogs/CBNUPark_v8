import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO


class ObjectDetection:

    def __init__(self, capture_index):
        # object detector init

        self.capture_index = capture_index

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Selected Device: ', self.device)

        self.model = self.load_model()


    def load_model(self):

        model = YOLO('weights/yolov8n.pt')
        model.fuse()

        return model
    
    def predict(self, frame):

        results = self.model(frame)

        return results
    
    def plot_bboxes(self, results, frame):

        xyxys = []
        confidences = []
        class_ids = []

        # Extract detections
        for result in results:
            boxes = result.boxes.cpu().numpy()

            xyxy = boxes.xyxy
            xywh = boxes.xywh

            print(boxes)
        
        return frame