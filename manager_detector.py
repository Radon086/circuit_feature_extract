from string import Template
import os
import cv2
import torch
from ultralytics import YOLO
import numpy as np

class DetectorManager:
    def __init__(self, yolo_checkpoint=None, device="cpu"):
        self.yolo_checkpoint = yolo_checkpoint if yolo_checkpoint is not None else "./model/yolo/yolov8_best.pt"
        self.device = device
        self.model = YOLO(self.yolo_checkpoint)
        self.model.to(self.device)

    def annotate_bounding_boxes(self,image_path, predictions, threshold=0.01): #0.4
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
    
    def comp_detection(self,image_path, yolo_checkpoint=None, device = "cpu"):
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