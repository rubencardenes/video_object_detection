from __future__ import annotations

from pathlib import Path
from threading import Event

import cv2
from loguru import logger
from PySide6.QtCore import QObject, QThread, Signal

from video_tools.core.image_sequence import ImageSequence, is_image_sequence
from video_tools.detection.config import DetectionConfig
from video_tools.detection.frame_utils import annotate_frame
from video_tools.detection.model import DetectionModel
from video_tools.detection.pipeline import DetectionTrackingPipeline


class DetectionSaveJob(QObject):
    """Annotate a video in a worker thread and write it as an MP4 file."""

    progress = Signal(float)  # 0.0-100.0
    finished = Signal(bool, str)  # success, destination path or error message

    def __init__(
        self,
        source: Path,
        destination: Path,
        model: DetectionModel,
        config: DetectionConfig,
        image_sequence_fps: float = 30.0,
        image_cache_size: int = 24,
        log_frames: bool = False,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._source = Path(source)
        self._destination = Path(destination)
        self._model = model
        self._config = config
        self._image_sequence_fps = image_sequence_fps
        self._image_cache_size = image_cache_size
        self._log_frames = log_frames
        self._cancelled = Event()
        self._thread: QThread | None = None
        self._worker: _DetectionSaveWorker | None = None

    def start(self) -> None:
        if self._thread is not None:
            return

        self._cancelled.clear()
        thread = QThread(self)
        worker = _DetectionSaveWorker(
            self._source,
            self._destination,
            self._model,
            self._config,
            self._cancelled,
            self._image_sequence_fps,
            self._image_cache_size,
            self._log_frames,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.progress)
        worker.finished.connect(self._on_worker_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    def cancel(self) -> None:
        self._cancelled.set()

    def _on_worker_finished(self, success: bool, message: str) -> None:
        self._worker = None
        self._thread = None
        self.finished.emit(success, message)


class _DetectionSaveWorker(QObject):
    progress = Signal(float)
    finished = Signal(bool, str)

    def __init__(
        self,
        source: Path,
        destination: Path,
        model: DetectionModel,
        config: DetectionConfig,
        cancelled: Event,
        image_sequence_fps: float,
        image_cache_size: int,
        log_frames: bool,
    ) -> None:
        super().__init__()
        self._source = source
        self._destination = destination
        self._model = model
        self._config = config
        self._cancelled = cancelled
        self._image_sequence_fps = image_sequence_fps
        self._image_cache_size = image_cache_size
        self._log_frames = log_frames

    def run(self) -> None:
        capture: cv2.VideoCapture | None = None
        writer: cv2.VideoWriter | None = None
        partial = self._destination.with_name(f".{self._destination.stem}.part.mp4")
        try:
            self._destination.parent.mkdir(parents=True, exist_ok=True)
            sequence = (
                ImageSequence(
                    self._source,
                    cache_size=self._image_cache_size,
                    default_fps=self._image_sequence_fps,
                )
                if is_image_sequence(self._source)
                else None
            )
            if sequence is not None:
                first = sequence.frame(0)
                height, width = first.shape[:2]
                fps = sequence.fps
                frame_count = len(sequence)
            else:
                capture = cv2.VideoCapture(str(self._source))
                if not capture.isOpened():
                    raise RuntimeError(f"Could not open input video: {self._source}")
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if width <= 0 or height <= 0:
                raise RuntimeError("Input video has invalid dimensions")
            if fps <= 0:
                raise RuntimeError("Input video has an invalid frame rate")

            writer = cv2.VideoWriter(
                str(partial), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output video: {self._destination}")

            pipeline = DetectionTrackingPipeline(
                self._model, self._config, log_frames=self._log_frames
            )
            processed = 0
            while not self._cancelled.is_set():
                if sequence is not None:
                    if processed >= len(sequence):
                        break
                    frame_rgb = sequence.frame(processed)
                else:
                    assert capture is not None
                    ok, frame_bgr = capture.read()
                    if not ok:
                        break
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                detections = pipeline.process(frame_rgb)
                annotated_rgb = annotate_frame(frame_rgb, detections)
                writer.write(cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
                processed += 1
                if frame_count > 0:
                    self.progress.emit(min(100.0, processed / frame_count * 100.0))

            if self._cancelled.is_set():
                raise _Cancelled
            if processed == 0:
                raise RuntimeError("Input video contains no readable frames")

            if capture is not None:
                capture.release()
                capture = None
            writer.release()
            writer = None
            partial.replace(self._destination)
            self.progress.emit(100.0)
            self.finished.emit(True, str(self._destination))
        except _Cancelled:
            self.finished.emit(False, "Cancelled")
        except Exception as exc:  # noqa: BLE001 - report worker failures to the GUI
            logger.exception(f"Detection save failed: {exc}")
            self.finished.emit(False, str(exc))
        finally:
            if capture is not None:
                capture.release()
            if writer is not None:
                writer.release()
            if partial.exists():
                partial.unlink()


class _Cancelled(Exception):
    pass
