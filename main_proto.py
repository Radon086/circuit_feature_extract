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


# img_list = ['./imgs/app_circuits/L298_8.png','./imgs/app_circuits/L298_9_2.png','./imgs/app_circuits/L298_10.png','./imgs/app_circuits/tps5430_7-9.png']
# img_list = ['./imgs/app_circuits/L298_10.png']
img_list = ['./imgs/block_circuit_diagram/I298_1.png']
#img_list = ['./imgs/functional_block/adxl354.png','./imgs/functional_block/TJA1024.png','./imgs/functional_block/tps5430.png']


manager_detector = DetectorManager(yolo_checkpoint="./model/yolo/yolov8_best.pt")
manager_ocr = OCRManager(det_arch="db_resnet50", reco_arch="master", pretrained=True)
manager_vlm = VLMManager(model_path="./model/gemma-4/gemma-4-12B-it", output_dir="./output/test_proto")

out_dir = "./output/test_proto"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

meta_list = []
for img_path in img_list:
    meta = {}
    ocr_results = manager_ocr.predict(img_path, thresh= 0.3)
    ocr_lines = manager_ocr.get_lines(ocr_results)
    manager_ocr.draw_lines(img_path, ocr_lines, output_path= out_dir)
    componets = manager_detector.comp_detection(img_path, threshold=0.1)
    manager_detector.save_annotated_image(img_path, componets, output_path= out_dir)
    ic_bbox = manager_vlm.locate_IC(img_path, out_dir,save_filterd=True,save_annotated=True)
    respond_ask_pins = manager_vlm.ask_pins(img_path, ic_bbox)

    file_name = img_path.split('/')[-1]
    if '.' in file_name:
        file_name = file_name.rsplit('.', 1)[0]  # Remove the extension
         
    meta["image_path"] = img_path
    meta["txt"] = ocr_lines
    meta["components"] = componets
    meta["ic"] = ic_bbox
    meta['ic_connection'] = respond_ask_pins

    meta_list.append({file_name: meta})

with open(f"{out_dir}/meta.jsonl", 'w') as f:
    for item in meta_list:
        json.dump(item, f)
        f.write('\n')