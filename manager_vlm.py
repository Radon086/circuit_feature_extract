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

    def remove_model_thinktags(self, response):
        if response.endswith("<turn|>"):
            response = response[:-7]
        if "<channel|>" in response:
            index = response.find("<channel|>")
            response = response[index+10:]
        return response
    
    def locate_IC(self,imgs):
        dic = []
        for img_path in imgs:
            print(f'\nGemma processing img:{img_path}')
            prompt = [
                {
                    "role": "user", "content": [
                        {"type": "image", "image": img_path},
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
            response = self.remove_model_thinktags(response)
            print("Response:\n", response)
            
            file_name = img_path.split('/')[-1]

            response = response.splitlines()
            img = cv2.imread(img_path)
            boxes = []
            for box in response:
                # normalze to 1000x1000
                y1, x1, y2, x2 = map(int, box.split(','))
                
                h,w = img.shape[:2]
                y1 = int(y1 * h / 1000)
                y2 = int(y2 * h / 1000)
                x1 = int(x1 * w / 1000)
                x2 = int(x2 * w / 1000)
                boxes.append([x1, y1, x2, y2])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            dic.append({"img_path": img_path, "boxes": boxes})
            cv2.imwrite(f'{self.output_dir}/gemma_annotated_{file_name}', img)
        return dic
    
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
        response = self.remove_model_thinktags(response)
        print("Response:\n", response)  