from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Lazy re-exports (PEP 562): the heavy submodules pull in cv2/onnx/Qt, so we only
# import them on first attribute access. This keeps `from video_tools.detection.config
# import DetectionConfig` (used by config/settings.py) free of that runtime.
_EXPORTS = {
    "DetectionConfig": "video_tools.detection.config",
    "annotate_frame": "video_tools.detection.frame_utils",
    "frame_to_rgb_array": "video_tools.detection.frame_utils",
    "rgb_array_to_qimage": "video_tools.detection.frame_utils",
    "COCO_CLASSES": "video_tools.detection.model",
    "DEFAULT_MODELS_DIR": "video_tools.detection.model",
    "DetectionModel": "video_tools.detection.model",
    "find_onnx_model": "video_tools.detection.model",
    "list_detection_models": "video_tools.detection.model",
    "list_onnx_models": "video_tools.detection.model",
    "resolve_model_path": "video_tools.detection.model",
    "DetectionTrackingPipeline": "video_tools.detection.pipeline",
    "DetectionSaveJob": "video_tools.detection.save_job",
    "TRACKING_METHODS": "video_tools.detection.tracking",
    "available_methods": "video_tools.detection.tracking",
    "create_tracker": "video_tools.detection.tracking",
}

__all__ = sorted(_EXPORTS)

if TYPE_CHECKING:  # let type checkers/IDEs resolve the lazily-exported names
    from video_tools.detection.config import DetectionConfig as DetectionConfig
    from video_tools.detection.frame_utils import annotate_frame as annotate_frame
    from video_tools.detection.frame_utils import frame_to_rgb_array as frame_to_rgb_array
    from video_tools.detection.frame_utils import rgb_array_to_qimage as rgb_array_to_qimage
    from video_tools.detection.model import COCO_CLASSES as COCO_CLASSES
    from video_tools.detection.model import DEFAULT_MODELS_DIR as DEFAULT_MODELS_DIR
    from video_tools.detection.model import DetectionModel as DetectionModel
    from video_tools.detection.model import find_onnx_model as find_onnx_model
    from video_tools.detection.model import list_detection_models as list_detection_models
    from video_tools.detection.model import list_onnx_models as list_onnx_models
    from video_tools.detection.model import resolve_model_path as resolve_model_path
    from video_tools.detection.pipeline import DetectionTrackingPipeline as DetectionTrackingPipeline
    from video_tools.detection.save_job import DetectionSaveJob as DetectionSaveJob
    from video_tools.detection.tracking import TRACKING_METHODS as TRACKING_METHODS
    from video_tools.detection.tracking import available_methods as available_methods
    from video_tools.detection.tracking import create_tracker as create_tracker


def __getattr__(name: str):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)
