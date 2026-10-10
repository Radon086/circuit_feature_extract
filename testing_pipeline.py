from transformers import AutoProcessor, AutoModelForMultimodalLM, AutoModelForImageTextToText
from string import Template
import os
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import json

MODEL_PATH = "./model/gemma-4/gemma-4-12B-it"
model = AutoModelForMultimodalLM.from_pretrained(MODEL_PATH, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

find_circuit_region_prompt = """
This is a preprocessed circuit diagram. Image preprocessing has removed many thin strokes and redundant visual details, but it may also have partially removed the boundaries of the rectangular IC packages.

Your task is to identify the remaining rectangular IC packages and estimate the bounding box of each complete package.

Instructions:
- Look for the large rectangular structures representing IC packages.
- The rectangle boundaries may be incomplete, and some internal circuit symbols may remain visible.
- Do not treat small rectangles, isolated line segments, or external components as target IC packages.
- Exclude external wires and components from each bounding box.
- There may be multiple IC packages in the image. Return one bounding box for each detected package.
- Estimate the full outer boundary of each package, including portions that have been removed by preprocessing.

Return ONLY the bounding boxes, one per line, in this format:
y1,x1,y2,x2

Coordinates must be normalized to the range 0–1000, relative to the image height and width respectively.
Do not include explanations or any other text.
"""
find_IC_PIN_prompt = Template(
    """
    You are analyzing a circuit diagram for semantic extraction.

    The supplied bounding box identifies the target IC region in the original circuit image.

    Bounding box in [x1, y1, x2, y2] format:
    $box_str

    Analyze the target IC and its visible pin connections.

    For each pin or terminal, report:
    1. Pin number or label, if visible.
    2. Pin location on the IC boundary (top, bottom, left, or right).
    3. The wire or net visibly connected to it.
    4. Any connected external component or labeled signal, if identifiable.
    5. Confidence and visual evidence.

    Json Format(Slot/Type/Purpose):
    1. pin_number/string or null/Visible pin number or label
    2. location/string/	top, bottom, left, or right
    3. connected_net/string or null/Visible net label, such as +VS, C, or V_en
    4. connected_component/string or null/Connected external component, if identifiable
    5. confidence/string/high, medium, or low

    Rules:
    - Use the original image to inspect wires and labels around the IC boundary.
    - Trace connections only when supported by visible evidence.
    - Do not invent missing pin numbers, labels, or convnections.
    - If a connection cannot be determined from the image, mark it as uncertain.
    - Distinguish directly observed connections from inferred ones.

    Return the result as JSON.
    """
)

find_IC_PIN_prompt_no_grounding = """
    You are analyzing a circuit diagram for semantic extraction.
    Analyze the target IC and its visible pin connections.

    For each pin or terminal, report:
    1. Pin number or label, if visible.
    2. Pin location on the IC boundary (top, bottom, left, or right).
    3. The wire or net visibly connected to it.
    4. Any connected external component or labeled signal, if identifiable.
    5. Confidence and visual evidence.

    Rules:
    - Use the original image to inspect wires and labels around the IC boundary.
    - Trace connections only when supported by visible evidence.
    - Do not invent missing pin numbers, labels, or connections.
    - If a connection cannot be determined from the image, mark it as uncertain.
    - Distinguish directly observed connections from inferred ones.
    
    Return the result as JSON.
    """

def remove_model_thinktags(response):
    if response.endswith("<turn|>"):
        response = response[:-7]
    if "<channel|>" in response:
        index = response.find("<channel|>")
        response = response[index+10:]
    return response

def annotate_bounding_boxes(image_path, predictions, threshold=0.01): #0.4
    """
    Annotates an image with bounding boxes and labels based on the given predictions tensor.
    
    Parameters:
    - image_path: Path to the input image.
    - predictions: A tensor containing bounding box coordinates, confidence scores, and class labels.

    Returns:
    - annotated_image: The image annotated with bounding boxes and labels as a numpy ndarray.
    - final_predictions: List of bounding box coordinates, scores, and labels.
    """

    predictions = predictions.cpu().numpy()
    
    print(f"Total predictions: {len(predictions)}")
    print(f"Predictions (x1, y1, x2, y2, score, class_id): {predictions}")

    def calculate_iou(box1, box2):
        """Calculate the Intersection over Union (IoU) of two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate intersection area
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate areas of each box
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Calculate union area
        union_area = box1_area + box2_area - inter_area

        # Avoid division by zero
        if union_area == 0:
            return 0

        # Calculate IoU
        iou = inter_area / union_area
        return iou

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Extract classes and names for label mapping
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    names = ['AC_Source', 'BJT', 'Battery', 'Capacitor', 'Current_Source', 'DC_Source',  'Diode', 'Ground', 'Inductor', 'MOSFET', 'Resistor', 'Voltage_Source']
    
    final_predictions = []
    #print("Shape of image",image.shape)
    
    # Iterate over the predictions
    for i, prediction in enumerate(predictions):
        x1, y1, x2, y2, score, class_id = prediction
        class_id = int(class_id)
        label = names[class_id]

        # Convert bounding box coordinates to integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        if score < threshold:
            #if label in ['Capacitor']:
            #    print(f"Skipping {label} with confidence {score:.2f} below threshold {threshold}")
            #else:
            #    if label in ['MOSFET', 'Resistor'] and score < 0.2:
            #        print(f"Skipping {label} with confidence {score:.2f} below threshold {threshold}")
            continue  # Skip if the confidence score is less than 0.7

        # Check overlap with MOSFETs
        overlap_with_mosfet = False
        for j, other_pred in enumerate(predictions):
            if i == j:
                continue  # Skip self-comparison
            _, _, _, _, _, other_class_id = other_pred
            if int(other_class_id) == 9:  # Check only MOSFET class
                other_x1, other_y1, other_x2, other_y2 = map(int, other_pred[:4])
                iou = calculate_iou([x1, y1, x2, y2], [other_x1, other_y1, other_x2, other_y2])
                if iou > 0.3:
                    overlap_with_mosfet = True
                    break

            if overlap_with_mosfet:
                continue  # Skip if overlap with MOSFET is greater than 0.3

        # Add to final predictions list
        final_predictions.append([x1, y1, x2, y2, score, class_id])

        # Annotate the image with bounding box and label
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        #print("Coordinates from annotated_img:")
        #print(x1,y1,x2,y2)
        text = f'{label} ({score:.2f})'
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert the final_predictions list to the specified format
    final_predictions = np.array(final_predictions, dtype=object)
    tmp = []
    for prediction in final_predictions:
        x1, y1, x2, y2, score, class_id = prediction
        label = names[int(class_id)]
        region_img = cv2.imread(image_path)[y1:y2, x1:x2]
        tmp.append(
            {
                "label": label,
                "confidence": float(score),
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "region_img": region_img  # Store the cropped region image for wire line analysis shit
            }
        )
        
    return image, tmp

def locate_IC(imgs):
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

        inputs = processor.apply_chat_template(
            prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)


        input_len = inputs["input_ids"].shape[-1]

        # Generate output
        outputs = model.generate(**inputs, max_new_tokens=512, temperature = 0.1)
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        response = remove_model_thinktags(response)
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
        cv2.imwrite(f'./output/test_pipeline/gemma_annotated_{file_name}', img)
    return dic

def ask_pins(img, bbox):

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

    inputs = processor.apply_chat_template(
        prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    response = remove_model_thinktags(response)
    print("Response:\n", response)  

# A compared version of ask_pins that does not include grounding information in the prompt.
def ask_pins_no_grounding(img):
    prompt = find_IC_PIN_prompt_no_grounding

    prompt = [
        {
            "role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    response = remove_model_thinktags(response)
    print("Response:\n", response)


# The filtered by thickness imgs        
#img_path = ['./output/test_cv/Width_Filtered_Small_Areas_0.png','./output/test_cv/Width_Filtered_Small_Areas_1.png','./output/test_cv/Width_Filtered_Small_Areas_2.png','./output/test_cv/Width_Filtered_Small_Areas_3.png']
#IC_bbox=locate_IC(img_path)
#with open('./output/test_pipeline/IC_bbox.jsonl', 'w') as f:
#    for item in IC_bbox:
#        json.dump(item, f)
#        f.write('\n')

"""
IC_bbox = []
origin_path = ['./imgs/app_circuits/L298_8.png','./imgs/app_circuits/L298_9.png','./imgs/app_circuits/L298_10.png','./imgs/app_circuits/tps5430_7-9.png']
with open('./output/test_pipeline/IC_bbox.jsonl', 'r') as f:
    for line in f:
        IC_bbox.append(json.loads(line.strip()))

for path, bbox in zip(origin_path, IC_bbox):
    print(f'\nGemma processing img:{path}')
    img = cv2.imread(path)
    ask_pins(path, bbox['boxes'])
"""
    
IC_bbox = []
origin_path = ['./imgs/app_circuits/L298_8.png','./imgs/app_circuits/L298_9.png','./imgs/app_circuits/L298_10.png','./imgs/app_circuits/tps5430_7-9.png']
for path in origin_path:
    print(f'\nGemma processing img:{path}')
    img = cv2.imread(path)
    ask_pins_no_grounding(img)
    break