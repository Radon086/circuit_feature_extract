from transformers import AutoProcessor, AutoModelForMultimodalLM, AutoModelForImageTextToText
from string import Template
import os
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import json
from prompts import find_circuit_region_prompt, find_IC_PIN_prompt

DEFAULT_MODEL_PATH = "./model/gemma-4/gemma-4-12B-it"

class VLMManager:
    def __init__(self, model_path=None, output_dir=None):
        self.model_path = model_path if model_path is not None else DEFAULT_MODEL_PATH
        self.output_dir = output_dir if output_dir is not None else "./output/test_proto"
        self.model = AutoModelForMultimodalLM.from_pretrained(self.model_path, dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.output_dir = "./output/test_proto"
        self.max_new_tokens = 1024
        self.temperature = 0.1


    def _find_strokes(self, image_path):
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        _, binary = cv2.threshold(
            gray, 200, 255, cv2.THRESH_BINARY_INV
        )

        dist = cv2.distanceTransform(
            binary, cv2.DIST_L2, 5
        )

        stroke_width = 2 * dist
        values = stroke_width[binary > 0]
        median = np.median(values).item()
        print("Min:", values.min())
        print("Max:", values.max())
        print("Median:", np.median(values))
        print("90th percentile:", np.percentile(values, 90))
        print("95th percentile:", np.percentile(values, 95))

        return values, median

    def _filter_by_thickness(self, image_path, threshold=4.0):
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        _, binary = cv2.threshold(
            gray, 200, 255, cv2.THRESH_BINARY_INV
        )

        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        stroke_width = 2 * dist

        # Keep only pixels with estimated stroke width >= 4 px
        width_mask = np.zeros_like(binary)
        width_mask[stroke_width >= threshold] = 255

        gray_filtered = cv2.bitwise_and(cv2.bitwise_not(gray), width_mask)
        return width_mask, gray_filtered

    def _remove_small_areas(self, image, min_area=10):
        # Works with a binary image: background=0, foreground=255
        binary = np.where(image > 0, 255, 0).astype(np.uint8)

        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(binary, connectivity=8)

        filtered = np.zeros_like(binary)

        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]

            if area >= min_area:
                filtered[labels == i] = 255
        return filtered

    def _remove_model_thinktags(self, response):
        if response.endswith("<turn|>"):
            response = response[:-7]
        if "<channel|>" in response:
            index = response.find("<channel|>")
            response = response[index+10:]
        return response
    
    def locate_IC(self,img_path, output_path, save_filterd=False, save_annotated=True):
        _, median = self._find_strokes(img_path)
        width_mask, _ = self._filter_by_thickness(img_path, threshold=median)
        width_mask = self._remove_small_areas(width_mask, min_area=10)
        file_name = None
        if save_filterd or save_annotated:
            file_name = img_path.split('/')[-1]
            if '.' in file_name:
                file_name = file_name.rsplit('.', 1)[0]  # Remove the extension

        if save_filterd:
            print(f"Width filtered image saved to {output_path}/Width_Filtered_{file_name}.png")
            cv2.imwrite(f"{output_path}/Width_Filtered_{file_name}.png", width_mask)

        dic = []
        print(f'\nVLM Finding IC:{img_path}')
        prompt = [
            {
                "role": "user", "content": [
                    {"type": "image", "image": width_mask},
                    {"type": "text", "text": find_circuit_region_prompt}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)


        input_len = inputs["input_ids"].shape[-1]

        # Generate output
        outputs = self.model.generate(**inputs, max_new_tokens=512, temperature = 0.1)
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        response = self._remove_model_thinktags(response)
        #print("Response:\n", response)
        file_name = img_path.split('/')[-1]

        response = response.splitlines()
        img = cv2.imread(img_path)
        h,w = img.shape[:2]
        boxes = []
        for box in response:
            # normalze to 1000x1000
            y1, x1, y2, x2 = map(int, box.split(','))
            y1 = int(y1 * h / 1000)
            y2 = int(y2 * h / 1000)
            x1 = int(x1 * w / 1000)
            x2 = int(x2 * w / 1000)
            boxes.append([x1, y1, x2, y2])
            if save_annotated:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if save_annotated:
            print(f"Annotated image saved to {output_path}/IC_annotated_{file_name}.png")
            cv2.imwrite(f"{output_path}/IC_annotated_{file_name}.png", img)
        return boxes
    
    def ask_pins(self,img, bbox):
        box_str = ""

        for box in bbox:
            x1, y1, x2, y2 = box
            box_str += f"[{x1}, {y1}, {x2}, {y2}]\n"

            prompt = find_IC_PIN_prompt.substitute(box_str=box_str)
            prompt = [
                {
                    "role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            inputs = self.processor.apply_chat_template(
                prompt,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(self.model.device)

            input_len = inputs["input_ids"].shape[-1]

            # Generate output
            outputs = self.model.generate(**inputs, max_new_tokens= self.max_new_tokens, temperature=self.temperature)
            response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)
            response = self._remove_model_thinktags(response)
            print("Response:\n", response)  