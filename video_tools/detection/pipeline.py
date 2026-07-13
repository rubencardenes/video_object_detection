from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv
from loguru import logger

from video_tools.detection.config import DetectionConfig
from video_tools.detection.model import DetectionModel
from video_tools.detection.tracking import create_tracker


@dataclass
class _Track:
    """One active visual tracker plus the class/confidence to re-attach on output."""

    tracker: object  # a cv2 single-object tracker
    class_id: int
    confidence: float


def _empty_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=int),
    )


class DetectionTrackingPipeline:
    """Runs ONNX detection every N frames and cheap visual tracking in between.

    Call process(frame_rgb) once per frame. It runs the detector when there is
    nothing to track, when the configured interval has elapsed, or early when the
    visual trackers have lost too many of the objects from the last detection
    (< reacquire_pct survive). Otherwise it just advances the visual trackers,
    which never touch the ONNX model — that is where the compute saving comes from.

    Not thread-safe: the caller must serialize process() calls (one frame at a
    time), which PreviewPanel already does via its _detection_busy guard.
    """

    def __init__(self, model: DetectionModel, config: DetectionConfig):
        self._model = model
        self._config = config
        self._tracks: list[_Track] = []
        self._frames_since_detection = 0
        self._last_detection_count = 0

    def update_config(self, config: DetectionConfig) -> None:
        self._config = config
        self.reset()

    def set_model(self, model: DetectionModel) -> None:
        self._model = model
        self.reset()

    def reset(self) -> None:
        self._tracks = []
        self._frames_since_detection = 0
        self._last_detection_count = 0

    def process(self, frame_rgb: np.ndarray) -> sv.Detections:
        need_detect = (
            not self._tracks or self._frames_since_detection >= self._config.interval_frames
        )
        if not need_detect:
            detections = self._update_trackers(frame_rgb)
            survivors, last = len(detections), self._last_detection_count
            if last and survivors / last * 100 < self._config.reacquire_pct:
                logger.info(f"Tracker kept {survivors}/{last} objects; re-detecting early")
                need_detect = True

        if need_detect:
            detections = self._model.infer(frame_rgb, threshold=self._config.confidence)
            self._init_trackers(frame_rgb, detections)
            self._frames_since_detection = 0
            self._last_detection_count = len(detections)
            logger.info(f"Detect: {len(detections)} object(s)")
        else:
            self._frames_since_detection += 1
            logger.debug(f"Track: {len(detections)} object(s) (frame +{self._frames_since_detection})")

        return detections

    def _init_trackers(self, frame_rgb: np.ndarray, detections: sv.Detections) -> None:
        height, width = frame_rgb.shape[:2]
        self._tracks = []
        confidence = detections.confidence
        class_id = detections.class_id
        for i, box in enumerate(detections.xyxy):
            xywh = self._clamp_to_xywh(box, width, height)
            if xywh is None:
                continue
            tracker = create_tracker(self._config.method)
            try:
                tracker.init(frame_rgb, xywh)
            except Exception as exc:  # noqa: BLE001 - a bad box shouldn't abort the whole frame
                logger.warning(f"Tracker init failed: {exc}")
                continue
            self._tracks.append(
                _Track(
                    tracker=tracker,
                    class_id=int(class_id[i]) if class_id is not None else 0,
                    confidence=float(confidence[i]) if confidence is not None else 1.0,
                )
            )

    def _update_trackers(self, frame_rgb: np.ndarray) -> sv.Detections:
        xyxy: list[list[float]] = []
        confidences: list[float] = []
        class_ids: list[int] = []
        survivors: list[_Track] = []
        for track in self._tracks:
            try:
                ok, box = track.tracker.update(frame_rgb)
            except Exception:  # noqa: BLE001 - drop a tracker that errored, keep the rest
                ok, box = False, None
            if not ok:
                continue
            x, y, w, h = box
            xyxy.append([float(x), float(y), float(x + w), float(y + h)])
            confidences.append(track.confidence)
            class_ids.append(track.class_id)
            survivors.append(track)
        self._tracks = survivors
        if not xyxy:
            return _empty_detections()
        return sv.Detections(
            xyxy=np.array(xyxy, dtype=np.float32),
            confidence=np.array(confidences, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int),
        )

    @staticmethod
    def _clamp_to_xywh(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = box
        x1 = int(max(0, min(x1, width - 1)))
        y1 = int(max(0, min(y1, height - 1)))
        x2 = int(max(0, min(x2, width)))
        y2 = int(max(0, min(y2, height)))
        w, h = x2 - x1, y2 - y1
        if w < 1 or h < 1:
            return None
        return x1, y1, w, h
