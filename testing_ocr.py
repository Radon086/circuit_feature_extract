from doctr.io import DocumentFile
from doctr.models import ocr_predictor


img_path = ['./imgs/app_circuits/L298_8.png','./imgs/app_circuits/L298_9.png','./imgs/app_circuits/L298_10.png','./imgs/app_circuits/tps5430_7-9.png']
model = ocr_predictor(det_arch="db_resnet50", reco_arch="master", pretrained=True)
thresh = 0.5
for image_path in img_path:
    doc = DocumentFile.from_images(image_path)
    model.det_predictor.model.postprocessor.box_thresh = thresh
    result = model(doc)

    print(f"\n===== threshold {thresh} =====")
    print(result.export())
    result.show()
