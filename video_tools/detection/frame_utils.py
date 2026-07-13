from __future__ import annotations

import numpy as np
import supervision as sv
from PySide6.QtGui import QImage
from PySide6.QtMultimedia import QVideoFrame

from video_tools.detection.model import COCO_CLASSES

_box_annotator = sv.BoxAnnotator()
_label_annotator = sv.LabelAnnotator()


def annotate_frame(frame_rgb: np.ndarray, detections: sv.Detections) -> np.ndarray:
    labels = [COCO_CLASSES.get(int(class_id), str(class_id)) for class_id in detections.class_id]
    annotated = _box_annotator.annotate(frame_rgb.copy(), detections)
    annotated = _label_annotator.annotate(annotated, detections, labels)
    return annotated


def frame_to_rgb_array(frame: QVideoFrame) -> np.ndarray | None:
    """Convert a QVideoFrame (as delivered by QVideoSink.videoFrameChanged) to an RGB ndarray."""
    image = frame.toImage()
    if image.isNull():
        return None
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    width, height, bytes_per_line = image.width(), image.height(), image.bytesPerLine()
    arr = np.frombuffer(image.constBits(), dtype=np.uint8, count=image.sizeInBytes())
    arr = arr.reshape((height, bytes_per_line))[:, : width * 3].reshape((height, width, 3))
    return arr.copy()  # detach from the QImage's buffer before it's garbage collected


def rgb_array_to_qimage(frame_rgb: np.ndarray) -> QImage:
    frame_rgb = np.ascontiguousarray(frame_rgb)
    height, width, _ = frame_rgb.shape
    image = QImage(frame_rgb.data, width, height, frame_rgb.strides[0], QImage.Format.Format_RGB888)
    return image.copy()  # detach from frame_rgb's buffer before it's garbage collected
