from transformers import AutoProcessor, AutoModelForMultimodalLM, AutoModelForImageTextToText
from string import Template
import os
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import json

"""
1. get a pic
2. run ocr scan, get the text and trivial stuffs.
3. run object detection. get components and their locations.
4. locate the IC.
5. Loctate the poins and get the connection info.
6. pack the info into a json file. 
"""

