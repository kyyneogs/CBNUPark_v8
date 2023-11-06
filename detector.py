import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO


class ObjectDetection:

    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model()
        print('Selected Device: ', self.device)
    