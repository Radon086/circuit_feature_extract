from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image
from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor

img_path = ['./imgs/app_circuits/L298_8.png','./imgs/app_circuits/L298_9.png','./imgs/app_circuits/L298_10.png','./imgs/app_circuits/tps5430_7-9.png']
model = ocr_predictor(det_arch="db_resnet50", reco_arch="master", pretrained=True)
thresh = 0.5

def test_with_doctr():
    for image_path in img_path:
        doc = DocumentFile.from_images(image_path)
        model.det_predictor.model.postprocessor.box_thresh = thresh
        result = model(doc)

        print(f"\n===== threshold {thresh} =====")
        print(result.export())
        result.show()
def test_with_surya():
    def surya_test(IMAGE_PATH):
        print(f"Running Surya OCR on {IMAGE_PATH}...")
        manager = SuryaInferenceManager()
        recognition_predictor = RecognitionPredictor(manager)
        
        # Default: full-page OCR. One VLM call per page. Returns one PageOCRResult per
        # image: `.blocks` (each with label, html, polygon, bbox, confidence, ...) and
        # `.image_bbox` — the same schema as block mode.
        predictions = recognition_predictor([Image.open(IMAGE_PATH)])
        '''
        # Block mode: pre-run layout, then per-block OCR. Same return schema as above.
        # Auto-selected when `layout_results` is passed.
        from surya.layout import LayoutPredictor
        layout = LayoutPredictor(manager)
        layouts = layout([Image.open(IMAGE_PATH)])
        predictions = recognition_predictor([Image.open(IMAGE_PATH)], layouts)
        '''
        return predictions
    
    for image_path in img_path:
        predictions = surya_test(image_path)
        for prediction in predictions:
            print(f"\n===== Surya OCR Result for {image_path} =====")
            print(prediction.export())
            
test_with_surya()