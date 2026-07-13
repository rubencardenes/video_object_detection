import cv2
import numpy as np
import supervision as sv
from rfdetr.assets.coco_classes import COCO_CLASSES
from rfdetr.export._onnx.inference import _create_onnx_session, _run_inference
import glob 
import time 

onnx_path = "models/rfdetr-medium.onnx"
images = glob.glob("/Users/ruben/Pictures/Seleccion2/*.JPG")

session = _create_onnx_session(onnx_path, providers=["CPUExecutionProvider"])
# session = _create_onnx_session(onnx_path, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
tv = []
for image in images:
    t = time.time()
    detections, pil_img = _run_inference(session, image, threshold=0.5)
    tv.append(time.time() - t)
    labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

    source_image = np.array(pil_img.convert("RGB"))
    annotated_image = sv.BoxAnnotator().annotate(source_image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

    cv2.namedWindow("result")
    cv2.imshow("result", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

print("Avg time ", np.mean(np.array(tv)))
