from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtGui import QImage

from video_tools.core.image_sequence import ImageSequence, is_image_sequence


def extract_thumbnail(
    path: Path, timestamp_sec: float = 1.0, size: tuple[int, int] = (160, 90)
) -> QImage | None:
    if is_image_sequence(path):
        sequence = ImageSequence(path, cache_size=1)
        frame = cv2.cvtColor(sequence.frame(0), cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        return QImage(
            frame.data, width, height, frame.strides[0], QImage.Format.Format_RGB888
        ).copy()
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
        ok, frame = cap.read()
        if not ok:
            # short clip: the requested timestamp was past the end, fall back to frame 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            return None

        frame = cv2.resize(frame, size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        image = QImage(frame.data, width, height, frame.strides[0], QImage.Format.Format_RGB888)
        return image.copy()  # detach from frame's buffer before it's garbage collected
    finally:
        cap.release()
