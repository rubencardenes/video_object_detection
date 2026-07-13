from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# ID -> COCO class name, vendored from rfdetr.assets.coco_classes (Apache-2.0)
# so this module only needs onnxruntime + supervision at runtime, not the full
# rfdetr/torch training stack.
COCO_CLASSES: dict[int, str] = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_THRESHOLD = 0.5


def list_onnx_models(models_dir: Path = DEFAULT_MODELS_DIR) -> list[str]:
    """Return the filenames of every *.onnx model in models_dir (for the UI dropdown)."""
    if not models_dir.is_dir():
        return []
    return [p.name for p in sorted(models_dir.glob("*.onnx"))]


def find_onnx_model(models_dir: Path = DEFAULT_MODELS_DIR) -> Path | None:
    """Return the first *.onnx file found in models_dir, or None if there isn't one."""
    if not models_dir.is_dir():
        return None
    matches = sorted(models_dir.glob("*.onnx"))
    return matches[0] if matches else None


def resolve_model_path(filename: str | None, models_dir: Path = DEFAULT_MODELS_DIR) -> Path | None:
    """Resolve a configured model filename to a path, falling back to the first available model."""
    if filename:
        candidate = models_dir / filename
        if candidate.is_file():
            return candidate
    return find_onnx_model(models_dir)


class DetectionModel:
    """An RF-DETR ONNX Runtime session plus the pre/post-processing to run it on one frame.

    Mirrors rfdetr.export._onnx.inference's preprocessing/decoding (BILINEAR
    resize, ImageNet normalisation, per-class sigmoid) but takes RGB numpy
    arrays instead of file paths/PIL images, so only onnxruntime + supervision
    are needed at runtime — not rfdetr or torch.
    """

    def __init__(self, model_path: Path, providers: list[str] | None = None):
        if providers is None:
            preferred = ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            available = ort.get_available_providers()
            providers = [p for p in preferred if p in available] or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(model_path), providers=providers)

        input_meta = self._session.get_inputs()[0]
        self._input_name = input_meta.name
        _, self._channels, self._height, self._width = input_meta.shape

        output_names = [out.name for out in self._session.get_outputs()]
        self._boxes_idx = next((i for i, name in enumerate(output_names) if "dets" in name), 0)
        self._logits_idx = next((i for i, name in enumerate(output_names) if "labels" in name), 1)

    def _preprocess(self, frame_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame_rgb, (self._width, self._height), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        return np.expand_dims(arr, axis=0).astype(np.float32)  # (1, C, H, W)

    def infer(self, frame_rgb: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> sv.Detections:
        """Run detection on one RGB frame; returns pixel-space xyxy boxes."""
        height, width = frame_rgb.shape[:2]
        inp_tensor = self._preprocess(frame_rgb)
        raw_outputs = self._session.run(None, {self._input_name: inp_tensor})

        boxes_cwh = raw_outputs[self._boxes_idx][0]  # (Q, 4) normalised cxcywh
        # RF-DETR adds a +1 no-object slot to num_classes; drop it or class_id
        # can equal len(COCO_CLASSES) and blow up the label lookup.
        logits = raw_outputs[self._logits_idx][0, :, :-1]  # (Q, num_classes)

        # RF-DETR uses per-class sigmoid, not softmax.
        one = np.asarray(1, dtype=logits.dtype)
        scores_all = one / (one + np.exp(-logits.clip(-88, 88)))
        scores = scores_all.max(axis=-1)
        cls = scores_all.argmax(axis=-1)
        keep = scores > threshold

        cx, cy, bw, bh = boxes_cwh[keep].T
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        xyxy *= np.array([width, height, width, height], dtype=np.float32)

        return sv.Detections(xyxy=xyxy, confidence=scores[keep], class_id=cls[keep].astype(int))
