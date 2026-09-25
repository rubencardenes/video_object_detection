from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv

from video_tools.detection.tensorrt_backend import TensorRTSession

DEFAULT_MODELS_DIR = Path("~/data/models").expanduser()

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


def list_detection_models(models_dir: Path = DEFAULT_MODELS_DIR) -> list[str]:
    """List selectable ONNX and TensorRT detector files."""
    if not models_dir.is_dir():
        return []
    return sorted(p.name for p in models_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in {".onnx", ".engine", ".plan"})


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
        configured = Path(filename).expanduser()
        candidate = configured if configured.is_absolute() else models_dir / configured
        if candidate.is_file():
            return candidate
    onnx = find_onnx_model(models_dir)
    models = list_detection_models(models_dir)
    return onnx or (models_dir / models[0] if models else None)


class DetectionModel:
    """ONNX/TensorRT detector supporting RF-DETR and raw Ultralytics YOLO exports."""

    def __init__(self, model_path: Path, providers: list[str] | None = None):
        self.backend_name = "TensorRT" if model_path.suffix.lower() in {".engine", ".plan"} else "ONNX"
        if providers is None:
            preferred = ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            available = ort.get_available_providers()
            providers = [p for p in preferred if p in available] or ["CPUExecutionProvider"]
        self._session = (
            TensorRTSession(model_path) if self.backend_name == "TensorRT"
            else ort.InferenceSession(str(model_path), providers=providers)
        )

        self.last_inference_ms: float | None = None

        input_meta = self._session.get_inputs()[0]
        self._input_name = input_meta.name
        _, self._channels, self._height, self._width = input_meta.shape

        # Reuse device buffers for fixed-shape CUDA exports. Transfers stay outside
        # the model timer, matching Ultralytics' inference-only measurement.
        self._io_binding = None
        self._device_input = None
        self._device_outputs = []
        output_meta = self._session.get_outputs()
        if (
            "CUDAExecutionProvider" in self._session.get_providers()
            and all(isinstance(d, int) and d > 0 for d in input_meta.shape)
            and all(
                out.type == "tensor(float)"
                and all(isinstance(d, int) and d > 0 for d in out.shape)
                for out in output_meta
            )
            and input_meta.type == "tensor(float)"
        ):
            self._io_binding = self._session.io_binding()
            self._device_input = ort.OrtValue.ortvalue_from_shape_and_type(
                input_meta.shape, np.float32, "cuda", 0
            )
            self._io_binding.bind_ortvalue_input(self._input_name, self._device_input)
            for out in output_meta:
                value = ort.OrtValue.ortvalue_from_shape_and_type(
                    out.shape, np.float32, "cuda", 0
                )
                self._device_outputs.append(value)
                self._io_binding.bind_ortvalue_output(out.name, value)

        output_names = [out.name for out in output_meta]
        output_shapes = [out.shape for out in self._session.get_outputs()]
        self._model_type = self._detect_model_type(output_names, output_shapes)
        self._boxes_idx = next((i for i, name in enumerate(output_names) if "dets" in name), 0)
        self._logits_idx = next((i for i, name in enumerate(output_names) if "labels" in name), 1)

    @staticmethod
    def _detect_model_type(output_names: list[str], output_shapes: list[list]) -> str:
        if any("dets" in name for name in output_names) and any(
            "labels" in name for name in output_names
        ):
            return "rfdetr"
        if len(output_shapes) == 1 and len(output_shapes[0]) == 3:
            return "yolo"
        raise ValueError(
            f"Unsupported ONNX detector outputs: {list(zip(output_names, output_shapes))}"
        )

    def _preprocess_rfdetr(self, frame_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame_rgb, (self._width, self._height), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        return np.expand_dims(arr, axis=0).astype(np.float32)  # (1, C, H, W)

    def _preprocess_yolo(
        self, frame_rgb: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        height, width = frame_rgb.shape[:2]
        scale = min(self._width / width, self._height / height)
        resized_width, resized_height = round(width * scale), round(height * scale)
        resized = cv2.resize(
            frame_rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
        pad_x = (self._width - resized_width) // 2
        pad_y = (self._height - resized_height) // 2
        canvas = np.full((self._height, self._width, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        tensor = cv2.dnn.blobFromImage(
            canvas, scalefactor=1 / 255.0, swapRB=False, crop=False
        )
        return tensor, scale, (pad_x, pad_y)

    def infer(self, frame_rgb: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> sv.Detections:
        """Run detection on one RGB frame; return pixel-space xyxy boxes."""
        if self._model_type == "yolo":
            return self._infer_yolo(frame_rgb, threshold)
        return self._infer_rfdetr(frame_rgb, threshold)

    def _run_inference(self, tensor: np.ndarray) -> list[np.ndarray]:
        self.last_inference_ms = None
        tensor = np.ascontiguousarray(tensor)
        if isinstance(self._session, TensorRTSession):
            outputs = self._session.run(None, {self._input_name: tensor})
            self.last_inference_ms = self._session.last_inference_ms
            return outputs
        if self._io_binding is not None:
            self._device_input.update_inplace(tensor)
            self._io_binding.synchronize_inputs()
            started = perf_counter()
            self._session.run_with_iobinding(self._io_binding)
            self._io_binding.synchronize_outputs()
            self.last_inference_ms = (perf_counter() - started) * 1000
            return self._io_binding.copy_outputs_to_cpu()
        started = perf_counter()
        outputs = self._session.run(None, {self._input_name: tensor})
        self.last_inference_ms = (perf_counter() - started) * 1000
        return outputs

    def _infer_rfdetr(
        self, frame_rgb: np.ndarray, threshold: float
    ) -> sv.Detections:
        height, width = frame_rgb.shape[:2]
        inp_tensor = self._preprocess_rfdetr(frame_rgb)
        raw_outputs = self._run_inference(inp_tensor)

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

    def _infer_yolo(self, frame_rgb: np.ndarray, threshold: float) -> sv.Detections:
        height, width = frame_rgb.shape[:2]
        inp_tensor, scale, (pad_x, pad_y) = self._preprocess_yolo(frame_rgb)
        output = self._run_inference(inp_tensor)[0][0]
        predictions = output.T if output.shape[0] < output.shape[1] else output
        boxes_cwh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        scores = class_scores.max(axis=1)
        yolo_classes = class_scores.argmax(axis=1)
        keep = scores > threshold
        boxes_cwh, scores, yolo_classes = boxes_cwh[keep], scores[keep], yolo_classes[keep]
        if not len(boxes_cwh):
            return _empty_detections()

        cx, cy, bw, bh = boxes_cwh.T
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        # Suppress before clipping: clipping changes IoU for border detections.
        selected = _class_aware_nms(xyxy, scores, yolo_classes, iou_threshold=0.7)
        selected = selected[:1000]
        xyxy = xyxy[selected]
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)

        coco_ids = np.asarray(_YOLO_TO_COCO_ID, dtype=int)[yolo_classes[selected]]
        return sv.Detections(
            xyxy=xyxy.astype(np.float32),
            confidence=scores[selected].astype(np.float32),
            class_id=coco_ids,
        )


_YOLO_TO_COCO_ID = tuple(COCO_CLASSES)


def _empty_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=int),
    )


def _class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    # OpenCV performs greedy NMS in compiled code; isolate classes explicitly.
    boxes_xywh = boxes.copy()
    boxes_xywh[:, 2:] -= boxes_xywh[:, :2]
    kept: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        selected = cv2.dnn.NMSBoxes(
            boxes_xywh[indices].tolist(), scores[indices].tolist(), 0.0, iou_threshold
        )
        kept.extend(indices[np.asarray(selected, dtype=int).reshape(-1)].tolist())
    kept = np.asarray(kept, dtype=int)
    return kept[np.argsort(-scores[kept], kind="stable")]
