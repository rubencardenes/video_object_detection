from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QFormLayout, QMessageBox, QPushButton, QVBoxLayout, QWidget

from video_tools.config.settings import Settings
from video_tools.core.ffmpeg_runner import FfmpegNotFoundError, find_ffmpeg
from video_tools.core.jobs import FfmpegJob, Worker
from video_tools.core.metadata import VideoInfo, probe_video
from video_tools.core.naming import suffixed_output_path
from video_tools.gui import icons
from video_tools.gui.widgets.progress_widget import ProgressWidget


class FfmpegTaskPanel(QWidget):
    """Base class for Convert/Resize/FPS/Cut panels: form + Run button + progress.

    Each panel is shown in its own pop-up dialog (opened from PreviewPanel's tool
    row), so it is self-contained. Subclasses populate self.form_layout with their
    own fields and implement output_suffix() and build_command(src, dst, ffmpeg).
    """

    def __init__(self, settings: Settings, parent: QWidget | None = None):
        super().__init__(parent)
        self._settings = settings
        self._current_path: Path | None = None
        self._pending_ffmpeg: str | None = None

        self.form_layout = QFormLayout()
        self._run_button = QPushButton("Run")
        self._run_button.setIcon(icons.play_icon())
        self._run_button.clicked.connect(self.run)
        self._progress_widget = ProgressWidget()

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.addLayout(self.form_layout)
        layout.addWidget(self._run_button)
        layout.addWidget(self._progress_widget)

    def load_video(self, path: Path) -> None:
        self._current_path = path
        self._progress_widget.reset()

    def output_suffix(self) -> str:
        raise NotImplementedError

    def build_command(self, src: Path, dst: Path, ffmpeg: str) -> list[str]:
        raise NotImplementedError

    def job_duration_sec(self, info: VideoInfo) -> float:
        """Expected duration of the ffmpeg job's output, for progress-percent math.

        Defaults to the full source duration; Cut overrides this since its output
        is only the trimmed span, not the whole source.
        """
        return info.duration_sec

    def run(self) -> None:
        if self._current_path is None:
            return
        try:
            ffmpeg = find_ffmpeg(self._settings)
        except FfmpegNotFoundError as exc:
            QMessageBox.warning(self, "ffmpeg not found", str(exc))
            return

        self._pending_ffmpeg = ffmpeg
        self._run_button.setEnabled(False)
        worker = Worker(probe_video, self._current_path, self._settings.ffprobe_path)
        # Connect to a bound method (not a lambda) so Qt recognizes the receiver's
        # thread affinity and queues delivery onto the GUI thread; a plain lambda
        # has no QObject affinity and would run on the worker thread instead.
        worker.signals.result.connect(self._on_probe_result)
        worker.signals.error.connect(self._on_probe_error)
        QThreadPool.globalInstance().start(worker)

    def _on_probe_result(self, info: VideoInfo) -> None:
        self._start_job(info, self._pending_ffmpeg)

    def _start_job(self, info: VideoInfo, ffmpeg: str) -> None:
        src = self._current_path
        dst = suffixed_output_path(src, self.output_suffix(), self._settings.output_dir)
        cmd = self.build_command(src, dst, ffmpeg)

        job = FfmpegJob(cmd, self.job_duration_sec(info), parent=self)
        self._progress_widget.bind(job)
        job.finished.connect(lambda success, message: self._run_button.setEnabled(True))
        job.start()

    def _on_probe_error(self, message: str) -> None:
        self._run_button.setEnabled(True)
        QMessageBox.warning(self, "Could not read video", message)
