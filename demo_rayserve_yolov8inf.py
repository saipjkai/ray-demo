import os
import json

import numpy as np
import cv2

import requests
from starlette.requests import Request
from typing import Dict

import ray
from ray import serve

from ultralytics import YOLO 
import base64


@serve.deployment(route_prefix="/")
class YOLOv8:
    def __init__(self, img_bytes_encode, weights=os.path.join(os.getcwd(), "yolo/weights/yolov8n.pt")):
        # Initialize model state: could be very large neural net weights.
        self._weights = weights
        self._model = YOLO(self._weights)
        self.img_bytes_encode = img_bytes_encode
        self.img = self._preprocess(self.img_bytes_encode)

    def __call__(self, request: Request) -> Dict:
        result = self._model.predict(self.img) 
        return {"result": result[0]}

    def _preprocess(self, img_bytes_encode):
        img_bytes_decode = base64.b64decode(self.img_bytes_encode)
        img_bytes = np.frombuffer(img_bytes_decode, dtype=np.uint8)
        img = cv2.imdecode(img_bytes, flags=1)
        return img


if __name__ == "__main__":
    base_dir = os.getcwd()
    img_path = os.path.join(base_dir, "data/images/bus.jpg")
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
    img_bytes_encode = base64.b64encode(img_bytes)


    serve.run(YOLOv8.bind(img_bytes_encode=img_bytes_encode))


    print(
        requests.get(
            "http://localhost:8000/")
        )
