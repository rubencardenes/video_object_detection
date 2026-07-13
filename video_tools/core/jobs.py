from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Callable

from loguru import logger
from PySide6.QtCore import QObject, QProcess, QRunnable, Signal
from PySide6.QtGui import QImage

from video_tools.core.thumbnails import extract_thumbnail


class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)


class Worker(QRunnable):
    """Runs an arbitrary blocking callable on QThreadPool, emitting result/error signals."""

    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the GUI thread
            logger.exception(f"Background job failed: {exc}")
            self.signals.error.emit(str(exc))
        else:
            self.signals.result.emit(result)


class ThumbnailSignals(QObject):
    done = Signal(Path, object)  # path, QImage | None


class ThumbnailWorker(QRunnable):
    def __init__(self, path: Path, size: tuple[int, int]):
        super().__init__()
        self._path = path
        self._size = size
        self.signals = ThumbnailSignals()

    def run(self) -> None:
        image: QImage | None
        try:
            image = extract_thumbnail(self._path, size=self._size)
        except Exception as exc:  # noqa: BLE001 - a bad thumbnail shouldn't crash the pool
            logger.warning(f"Thumbnail extraction failed for {self._path}: {exc}")
            image = None
        self.signals.done.emit(self._path, image)


def _parse_out_time(value: str) -> float:
    # ffmpeg -progress "out_time=" lines look like "00:00:12.345678"
    try:
        h, m, s = value.strip().split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except ValueError:
        return 0.0


class FfmpegJob(QObject):
    """Runs an ffmpeg command via QProcess, emitting parsed progress and a final result."""

    progress = Signal(float)  # 0.0-100.0
    finished = Signal(bool, str)  # success, message (or error/stderr tail)

    def __init__(self, cmd: list[str], total_duration_sec: float, parent: QObject | None = None):
        super().__init__(parent)
        self._cmd = cmd
        self._total_duration_sec = total_duration_sec
        self._stderr_tail: deque[str] = deque(maxlen=40)

        self._process = QProcess(self)
        self._process.readyReadStandardError.connect(self._on_stderr)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error_occurred)

    def start(self) -> None:
        program, *args = self._cmd
        logger.info(f"Running: {' '.join(self._cmd)}")
        self._process.start(program, args)

    def cancel(self) -> None:
        self._process.kill()

    def _on_stderr(self) -> None:
        data = bytes(self._process.readAllStandardError()).decode(errors="replace")
        self._stderr_tail.append(data)
        for line in data.splitlines():
            line = line.strip()
            if line.startswith("out_time="):
                seconds = _parse_out_time(line.split("=", 1)[1])
                if self._total_duration_sec > 0:
                    percent = min(100.0, seconds / self._total_duration_sec * 100)
                    self.progress.emit(percent)
            elif line == "progress=end":
                self.progress.emit(100.0)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        success = exit_code == 0 and exit_status == QProcess.ExitStatus.NormalExit
        message = "Done" if success else "".join(self._stderr_tail)[-2000:]
        self.finished.emit(success, message)

    def _on_error_occurred(self, error: QProcess.ProcessError) -> None:
        if self._process.state() == QProcess.ProcessState.NotRunning:
            self.finished.emit(False, f"ffmpeg failed to start: {error}")
