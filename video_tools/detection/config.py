from __future__ import annotations

from dataclasses import dataclass

# Kept intentionally dependency-free (no cv2/onnx/Qt) so config/settings.py can
# import DetectionConfig without pulling in the heavy detection runtime stack.

DEFAULT_INTERVAL_FRAMES = 5
DEFAULT_METHOD = "CSRT"
DEFAULT_REACQUIRE_PCT = 50
DEFAULT_CONFIDENCE = 0.5


@dataclass
class DetectionConfig:
    """User-tunable detection/tracking parameters (persisted in settings.yaml)."""

    interval_frames: int = DEFAULT_INTERVAL_FRAMES  # frames tracked between detections
    method: str = DEFAULT_METHOD  # visual tracker: CSRT | KCF | MOSSE | MIL
    model_filename: str | None = None  # .onnx name in models dir; None -> first found
    reacquire_pct: int = DEFAULT_REACQUIRE_PCT  # re-detect when < this % of objects still tracked
    confidence: float = DEFAULT_CONFIDENCE  # ONNX detection threshold
