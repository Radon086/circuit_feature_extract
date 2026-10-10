from transformers import AutoProcessor, AutoModelForMultimodalLM, AutoModelForImageTextToText
from string import Template
import os
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import json
from manager_detector import DetectorManager
from manager_ocr import OCRManager
from manager_vlm import VLMManager


"""
1. get a pic
2. run ocr scan, get the text and trivial stuffs.
3. run object detection. get components and their locations.
4. locate the IC.
5. Loctate the poins and get the connection info.
6. pack the info into a json file. 
"""


img_path = ['./imgs/app_circuits/L298_8.png','./imgs/app_circuits/L298_9_2.png','./imgs/app_circuits/L298_10.png','./imgs/app_circuits/tps5430_7-9.png']
img_path = ['./imgs/app_circuits/L298_8.png']

manager_detector = DetectorManager(yolo_checkpoint="./model/yolo/yolov8_best.pt", device="cpu")
manager_ocr = OCRManager(det_arch="db_resnet50", reco_arch="master", pretrained=True)
manager_vlm = VLMManager(model_path="./model/gemma-4/gemma-4-12B-it", output_dir="./output/test_proto")

