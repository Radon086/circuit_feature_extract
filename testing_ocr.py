from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor
from PIL import Image, ImageDraw
from pathlib import Path
import re
import html


img_path = ['./imgs/app_circuits/L298_8.png','./imgs/app_circuits/L298_9_2.png','./imgs/app_circuits/L298_10.png','./imgs/app_circuits/tps5430_7-9.png']
#img_path = ['./imgs/app_circuits/L298_8.png']
model = ocr_predictor(det_arch="db_resnet50", reco_arch="master", pretrained=True)
thresh = 0.5

def merge_words_into_lines(words, y_tolerance=0.5):
    """
    words: [{"text": str, "bbox": [x1, y1, x2, y2]}, ...]

    y_tolerance: maximum normalized vertical-center difference,
                 relative to the average word height.
    """
    words = sorted(words, key=lambda w: (
        (w["bbox"][1] + w["bbox"][3]) / 2
    ))

    lines = []

    for word in words:
        x1, y1, x2, y2 = word["bbox"]
        cy = (y1 + y2) / 2
        height = max(1, y2 - y1)

        best_line = None
        best_distance = float("inf")

        for line in lines:
            avg_cy = line["cy"]
            avg_height = line["height"]
            distance = abs(cy - avg_cy) / max(height, avg_height)

            if distance <= y_tolerance and distance < best_distance:
                best_line = line
                best_distance = distance

        if best_line is None:
            lines.append({
                "words": [word],
                "cy": cy,
                "height": height
            })
        else:
            best_line["words"].append(word)
            n = len(best_line["words"])
            best_line["cy"] += (cy - best_line["cy"]) / n
            best_line["height"] += (height - best_line["height"]) / n

    merged = []

    for line in lines:
        line_words = sorted(line["words"], key=lambda w: w["bbox"][0])
        boxes = [w["bbox"] for w in line_words]

        merged.append({
            "text": " ".join(w["text"] for w in line_words),
            "bbox": [
                min(b[0] for b in boxes),
                min(b[1] for b in boxes),
                max(b[2] for b in boxes),
                max(b[3] for b in boxes)
            ]
        })

    return sorted(merged, key=lambda line: (
        line["bbox"][1], line["bbox"][0]
    ))

def test_with_doctr(imgs, thresh=0.5, output_dir="./output/test_ocr"):
    for image_path in imgs:
        doc = DocumentFile.from_images(image_path)
        model.det_predictor.model.postprocessor.box_thresh = thresh
        result = model(doc)

        print(f"\n===== threshold {thresh} =====")
        #print(result.export())
        pages = result.export()["pages"]
        #print(len(pages))
        # only one page is expected for a single image
        # page -> block -> line -> word
        for page_idx, page in enumerate(result.pages):
            for block in page.blocks:
                for line in block.lines:
                    text = " ".join(word.value for word in line.words)

                    boxes = [word.geometry for word in line.words]

                    x1 = min(box[0][0] for box in boxes)
                    y1 = min(box[0][1] for box in boxes)
                    x2 = max(box[1][0] for box in boxes)
                    y2 = max(box[1][1] for box in boxes)

                    print({
                        "text": text,
                        "bbox": [x1, y1, x2, y2]
                    })
        
        """
        meta = pages[0]
        print(meta.keys())
        blocks = meta["blocks"]
        print(len(blocks))
        """
        #rendered = result.render()
        #ouput_path = Path(output_dir) / f"{Path(image_path).stem}_doctr.png"
        #Image.fromarray(rendered).save(output_path)
        result.show()
        

surya_manager = SuryaInferenceManager()
surya_recognition_predictor = RecognitionPredictor(surya_manager)


def show_surya_result(image_path, prediction, output_dir="./output/test_ocr"):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for block in prediction.blocks:
        bbox = block.bbox  # [x1, y1, x2, y2], pixel coordinates

        if bbox is None or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)

        # Surya's block content may contain HTML.
        text = html.unescape(
            re.sub(r"<[^>]+>", " ", block.html or "")
        ).strip().replace("\n", " ")

        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)

        if text:
            draw.text((x1, max(0, y1 - 14)), text[:60], fill="blue")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{Path(image_path).stem}_surya.png"
    image.save(output_path)

    print(f"Saved visualization: {output_path}")
    return image

def test_with_surya():
    def surya_test(IMAGE_PATH):
        print(f"Running Surya OCR on {IMAGE_PATH}...")
        manager = surya_manager
        recognition_predictor = surya_recognition_predictor
        
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
    
    for path in img_path:
        predictions = surya_test(path)
        for prediction in predictions:
            print(f"\n===== Surya OCR Result for {path} =====")
            print(prediction)
            show_surya_result(path, prediction)

# test_with_surya()
test_with_doctr(imgs=img_path,thresh=0.3)