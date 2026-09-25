from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class MotDetection:
    track_id: int
    left: float
    top: float
    width: float
    height: float
    confidence: float
    class_id: int


def load_mot_detections(path: Path) -> dict[int, list[MotDetection]]:
    """Parse MOTChallenge detections, keyed by the format's 1-based frame number."""
    detections: dict[int, list[MotDetection]] = {}
    with Path(path).open(newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), start=1):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 7:
                raise ValueError(
                    f"Invalid MOT row {line_number} in {path}: expected 7+ columns"
                )
            try:
                frame_number = int(float(row[0]))
                detection = MotDetection(
                    track_id=int(float(row[1])),
                    left=float(row[2]),
                    top=float(row[3]),
                    width=float(row[4]),
                    height=float(row[5]),
                    confidence=float(row[6]),
                    class_id=int(float(row[7])) if len(row) > 7 else -1,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid MOT values on row {line_number} in {path}"
                ) from exc
            detections.setdefault(frame_number, []).append(detection)
    return detections


def annotate_mot_frame(
    frame_rgb: np.ndarray, detections: list[MotDetection], confidence: float = 0.0
) -> np.ndarray:
    annotated = frame_rgb.copy()
    for detection in detections:
        if detection.confidence < confidence:
            continue
        x1, y1 = round(detection.left), round(detection.top)
        x2 = round(detection.left + detection.width)
        y2 = round(detection.top + detection.height)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 170, 0), 2)
        identity = f"ID {detection.track_id} " if detection.track_id >= 0 else ""
        label = f"{identity}{detection.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 170, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated
