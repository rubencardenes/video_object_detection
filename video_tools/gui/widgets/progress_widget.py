from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QWidget

from video_tools.core.jobs import FfmpegJob
from video_tools.gui import icons


class ProgressWidget(QWidget):
    """Thin progress row that is hidden while idle and only appears during a job.

    Keeping it hidden when there's nothing running removes the always-present
    "0 %" bar and saves vertical space; the percentage shows in the status label
    (not baked into the bar) so the bar itself can stay slim.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._job: FfmpegJob | None = None

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(6)
        self._status_label = QLabel()
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setIcon(icons.cancel_icon())
        self._cancel_button.clicked.connect(self._on_cancel)
        self._cancel_button.setEnabled(False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._bar, stretch=1)
        layout.addWidget(self._status_label)
        layout.addWidget(self._cancel_button)

        self.reset()

    def reset(self) -> None:
        self._bar.setValue(0)
        self._status_label.setText("")
        self._cancel_button.setEnabled(False)
        self.hide()  # take up no space until a job actually runs

    def bind(self, job: FfmpegJob) -> None:
        self._job = job
        self._bar.setValue(0)
        self._status_label.setText("0%")
        self._cancel_button.setEnabled(True)
        self.show()
        job.progress.connect(self._on_progress)
        job.finished.connect(self._on_finished)

    def _on_progress(self, percent: float) -> None:
        self._bar.setValue(int(percent))
        self._status_label.setText(f"{int(percent)}%")

    def _on_finished(self, success: bool, message: str) -> None:
        self._cancel_button.setEnabled(False)
        if success:
            self._bar.setValue(100)
            self._status_label.setText("Done")
        else:
            self._status_label.setText(f"Failed: {message[-200:]}")
        self._job = None

    def _on_cancel(self) -> None:
        if self._job is not None:
            self._job.cancel()
            self._status_label.setText("Cancelled")
            self._cancel_button.setEnabled(False)
