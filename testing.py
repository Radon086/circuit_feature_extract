from transformers import AutoProcessor, AutoModelForMultimodalLM, AutoModelForImageTextToText
from string import Template
import os
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image

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

test_prompt = """
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
Do not include explanations, your reasoning or any other text.
"""


def gemma_test(imgs):
    MODEL_PATH = "./model/gemma-4/gemma-4-12B-it"
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_PATH, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    img_path = './imgs/app_circuits/L298_8.png'
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {os.path.abspath(img_path)}")
    else:
        print('Image path loaded.')

    txt_prompt_template = Template("""
    Locate the $target integrated circuit bounding box in the image.

    Return ONLY in this format:
    y1,x1,y2,x2

    The coordinates must be normalized to 0-1000.
    
    Do not include surrounding wires, external components, or the rest of the circuit.
    Do not explain.
    Do not describe the circuit.
    Do not output any other text.
    """)

    '''
    txt_prompt = """
        ### Task:
        Locate the cat and return its bounding box.

        Return ONLY four integers:
        y1, x1, y2, x2

        Do not explain.
        Do not describe the circuit.
        Do not output any other text.
    """
    '''
    
    for img_path in imgs:
        print(f'\nGemma processing img:{img_path}')

        target = ''
        if "L298" in img_path:
            target = "L298"
        elif "tps5430" in img_path:
            target = "TPS5430"

        #txt_prompt = txt_prompt_template.substitute(target=target)
        txt_prompt = ''
        if target == "L298":
            txt_prompt = """

                            Find the big rectangulars in the picture.

                            Return ONLY:
                            y1,x1,y2,x2
            """
        elif target == "TPS5430":
            txt_prompt = """
            Locate the TPS5430 integrated circuit.

            Find the rectangular device package labeled TPS5430DDA.
            Return the bounding box of the complete IC package.

            Do not include surrounding wires or external components.

            Return ONLY:
            y1,x1,y2,x2

            Coordinates are normalized from 0 to 1000.
            """
        else:
            txt_prompt = """

                This is a filtered circuit diagram. The filter tried to filter redundent features rather than the appilcation circuit, which is represent by  rectangulars. 
                However, the filter is not perfect enough to leave only the rectanguler on the picture.
                Support the filter's task. Judge the rectangulars on the picture and return their possible locations.
                There can be more than one rectangulars in the picture.

                Return ONLY in a string of four integers:
                y1,x1,y2,x2
            """
            txt_prompt = test_prompt
        prompt = [
            {
                "role": "user", "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": txt_prompt}
                ]
            }
        ]

        def remove_model_thinktags(response):
            if response.endswith("<turn|>"):
                response = response[:-7]
            if "<channel|>" in response:
                index = response.find("<channel|>")
                response = response[index+10:]
            return response

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
        for box in response:
            # normalze to 1000x1000
            y1, x1, y2, x2 = map(int, box.split(','))
            
            h,w = img.shape[:2]
            y1 = int(y1 * h / 1000)
            y2 = int(y2 * h / 1000)
            x1 = int(x1 * w / 1000)
            x2 = int(x2 * w / 1000)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(f'./output/test/gemma_annotated_{file_name}', img)
        

def yolo_test(image_path):

    def comp_detection(image_path, yolo_checkpoint=None, device = "cpu"):
        print(f"⏳ Analyzing circuit image...")
        if yolo_checkpoint is None:
            yolo_checkpoint="./model/yolo/yolov8_best.pt"
            print(f"Using default vision model checkpoint: {yolo_checkpoint}")
        # Load the pretrained YOLO model
        model = YOLO(yolo_checkpoint)
        model.to(device)

        if not os.path.exists(image_path):
                print(f"Error: The image {image_path} does not exist.")
        try:
            # Perform prediction
            result = model.predict(source=image_path, verbose = False, iou=0.4)
            print("✓ Vision features extracted.")
            annotated_img, prediction = annotate_bounding_boxes(image_path,result[0].boxes.data)

            
            return annotated_img, prediction
                
        except Exception as e:
            print(f"An error occurred while processing {image_path}: {e}")

    img = cv2.imread(image_path)
    crossing_lines = []
    source_and_ground = []
    
    if torch.cuda.is_available():
        device = 'cuda'
    
    # run component detection
    annotated_img, components = comp_detection(image_path, device=device) # yolo takes image path
    print(f"Detected components: {len(components)}")
    '''
    for comp in components:
        # print(f"Component: {comp['label']}, Box: {comp['box']}, Confidence: {comp['confidence']:.2f}")
        if comp['label'] in ['Ground']:
            source_and_ground.append(['ground', comp['box']])
    '''
    return annotated_img, components
    
def qwen_test(imgs):
    model_path = "./model/qwen3vl/"
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(model_path)

    txt_prompt = """
        ### Task:
        Locate the cat and return its bounding box.

        Return ONLY in this format:
        x1, y1, x2, y2

        Do not explain.
        Do not describe the circuit.
        Do not output any other text.
    """
    
    txt_prompt = """
        ### Task:
        Locate the rectangles labeled with l289 or tps5430 and return its bounding box.

        Return ONLY in this format:
        x1, y1, x2, y2

        Do not explain.
        Do not describe the circuit.
        Do not output any other text.
    """
    

    for img_path in imgs:
        print(f'\nQwen processing img:{img_path}')
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": txt_prompt},
                ],
            }
        ]

        # Preparation for inference
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)    

        file_name = img_path.split('/')[-1]
        response = output_text[0]
        response = response.split('/n')
        img = cv2.imread(img_path)
        for box in response:
            # normalze to 1000x1000
            x1, y1, x2, y2 = map(int, box.split(','))
            
            h,w = img.shape[:2]
            y1 = int(y1 * h / 1000)
            y2 = int(y2 * h / 1000)
            x1 = int(x1 * w / 1000)
            x2 = int(x2 * w / 1000)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(f'./output/test/qwen_annotated_{file_name}', img)
    

img_path = ['./imgs/app_circuits/L298_8.png','./imgs/app_circuits/L298_9.png','./imgs/app_circuits/L298_10.png','./imgs/app_circuits/tps5430_7-9.png']
img_path = ['./imgs/app_circuits/L298_8.png','./imgs/app_circuits/L298_9.png','./imgs/app_circuits/L298_10.png']
#img_path = ['./imgs/app_circuits/cat.png']
#img_path = './imgs/app_circuits/2-stage-5tota.png'
img_path = ['./output/test_cv/Width_Filtered_Small_Areas_0.png','./output/test_cv/Width_Filtered_Small_Areas_1.png','./output/test_cv/Width_Filtered_Small_Areas_2.png','./output/test_cv/Width_Filtered_Small_Areas_3.png','./output/test_cv/Width_Filtered_Small_Areas_4.png']

if not os.path.exists('./output/test'):
    os.makedirs('./output/test')

#qwen_test(img_path)
gemma_test(img_path)


'''
id = 1    
for path in img_path:
    annotated_img, components = yolo_test(path)
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    annotated_img_rgb = Image.fromarray(annotated_img_rgb)
    annotated_img_rgb.save(f'./output/test/annotated_img{id}.png')
    id +=1
'''
#with open('./output/test/annotated_img.png', 'wb') as f:
#    cv2.imwrite('annotated_img.png', annotated_img)