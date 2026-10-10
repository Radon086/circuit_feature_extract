from doctr.io import DocumentFile
from doctr.models import ocr_predictor

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

                    x1 = min(box[0][0] for box in boxes)
                    y1 = min(box[0][1] for box in boxes)
                    x2 = max(box[1][0] for box in boxes)
                    y2 = max(box[1][1] for box in boxes)

                    ocr_results.append({
                        "text": text,
                        "bbox": [x1, y1, x2, y2]
                    })
        
        return ocr_results