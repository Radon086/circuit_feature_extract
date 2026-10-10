from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import cv2

class OCRManager:
    def __init__(self, det_arch="db_resnet50", reco_arch="master", pretrained=True):
        self.model = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=pretrained)


    def predict(self, image_path, thresh=0.5):
        doc = DocumentFile.from_images(image_path)
        self.model.det_predictor.model.postprocessor.box_thresh = thresh
        result = self.model(doc)
        return result
    
    def get_lines(self, result):
        ocr_results = []
        for page_idx, page in enumerate(result.pages):
            for block in page.blocks:
                for line in block.lines:
                    text = " ".join(word.value for word in line.words)
                    boxes = [word.geometry for word in line.words]

                    x1 = min(box[0][0] for box in boxes).item()
                    y1 = min(box[0][1] for box in boxes).item()
                    x2 = max(box[1][0] for box in boxes).item()
                    y2 = max(box[1][1] for box in boxes).item()

                    ocr_results.append({
                        "text": text,
                        "bbox": [x1, y1, x2, y2] # at portion of the image where the text is located
                    })
        return ocr_results
    
    def draw_lines(self, image_path, ocr_results, output_path):
        image = cv2.imread(image_path)
        file_name = image_path.split('/')[-1]
        if '.' in file_name:
            file_name = file_name.rsplit('.', 1)[0]  # Remove the extension

        h, w = image.shape[:2]
        for line in ocr_results:
            x1, y1, x2, y2 = line["bbox"]
            x1 = int(x1 * w)
            x2 = int(x2 * w)
            y1 = int(y1 * h)
            y2 = int(y2 * h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, line["text"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(f'{output_path}/ocr_annotated_{file_name}.png', image)
        print(f"OCR annotated image saved to {output_path}/ocr_annotated_{file_name}.png")