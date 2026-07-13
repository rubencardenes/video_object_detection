from __future__ import annotations

from pathlib import Path

from loguru import logger
from PySide6.QtCore import QThreadPool, QUrl, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoFrame
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStackedLayout,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from video_tools.core.detection import (
    DEFAULT_MODELS_DIR,
    DetectionModel,
    annotate_frame,
    find_onnx_model,
    frame_to_rgb_array,
    rgb_array_to_qimage,
)
from video_tools.core.jobs import Worker
from video_tools.gui import icons
from video_tools.gui.widgets.trim_range_bar import TrimRangeBar

# Tool buttons that swap the panel below them. Info is deliberately not one of
# these: it opens in its own popup dialog instead (see info_clicked below).
TOOL_NAMES = ("Convert", "Cut", "Resize", "FPS")
TOOL_ICON_FUNCS = {
    "Convert": icons.convert_icon,
    "Cut": icons.cut_icon,
    "Resize": icons.resize_icon,
    "FPS": icons.fps_icon,
}


def _format_ms(ms: int) -> str:
    total_seconds = ms // 1000
    m, s = divmod(total_seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class PreviewPanel(QWidget):
    """Constant chrome shown once a video is selected: player + action buttons.

    The tool-specific content below the buttons swaps via an internal
    QStackedWidget; panels are registered from outside via register_panel()
    so each milestone can plug its panel in without PreviewPanel knowing
    about ffmpeg/cut/resize specifics.
    """

    info_clicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_output)
        self._video_widget = QVideoWidget()
        self._video_widget.setMinimumHeight(280)
        # QVideoWidget does its own native/GPU video compositing. An app-wide
        # stylesheet auto-enables stylesheet background painting on every widget
        # (including this one), which fights that native paint path and can crash
        # on some platforms/backends. Opt this widget out explicitly.
        self._video_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self._player.setVideoOutput(self._video_widget)

        # Detection overlay: a QLabel that SWAPS IN for the video widget while
        # detecting (QStackedLayout's default StackOne mode - only one of the
        # two is ever the "current" widget). QVideoWidget does its own
        # native/GPU compositing that ignores normal Qt sibling z-order, so a
        # label simply stacked on top of it (StackAll) never actually became
        # visible - swapping instead of layering sidesteps that entirely.
        self._detection_overlay = QLabel()
        self._detection_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_container = QWidget()
        self._video_stack = QStackedLayout(self._video_container)
        self._video_stack.addWidget(self._video_widget)
        self._video_stack.addWidget(self._detection_overlay)
        self._video_stack.setCurrentWidget(self._video_widget)

        self._detection_model: DetectionModel | None = None
        self._detection_model_load_attempted = False
        self._detection_busy = False
        self._detection_sink_connected = False
        self._resume_after_detection = False
        self._detection_frame_index = 0

        self._play_button = QPushButton("Play")
        self._play_button.setIcon(icons.play_icon())
        self._play_button.clicked.connect(self._toggle_playback)

        self._detect_button = QPushButton("Detect")
        self._detect_button.setIcon(icons.detect_icon())
        self._detect_button.setCheckable(True)
        self._detect_button.toggled.connect(self._on_detect_toggled)

        self._position_slider = QSlider(Qt.Orientation.Horizontal)
        # sliderMoved only fires while actively dragging the handle; a plain click
        # on the groove moves the slider's value but never emits it, so without
        # sliderReleased the player position never actually changes and the next
        # positionChanged update snaps the handle back to the old position.
        self._position_slider.sliderMoved.connect(self._player.setPosition)
        self._position_slider.sliderReleased.connect(self._on_slider_released)

        self._trim_range_bar = TrimRangeBar()

        self._time_label = QLabel("00:00 / 00:00")

        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)

        slider_column = QVBoxLayout()
        slider_column.setSpacing(2)
        slider_column.addWidget(self._position_slider)
        slider_column.addWidget(self._trim_range_bar)

        controls_row = QHBoxLayout()
        controls_row.addWidget(self._play_button)
        controls_row.addWidget(self._detect_button)
        controls_row.addLayout(slider_column, stretch=1)
        controls_row.addWidget(self._time_label)

        self._tool_buttons: dict[str, QPushButton] = {}
        self._tool_button_group = QButtonGroup(self)
        self._tool_button_group.setExclusive(True)
        button_row = QHBoxLayout()

        info_button = QPushButton("Info")
        info_button.setIcon(icons.info_icon())
        info_button.clicked.connect(self.info_clicked)
        button_row.addWidget(info_button)

        for name in TOOL_NAMES:
            button = QPushButton(name)
            button.setIcon(TOOL_ICON_FUNCS[name]())
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, n=name: self.show_tool(n))
            button_row.addWidget(button)
            self._tool_buttons[name] = button
            self._tool_button_group.addButton(button)

        self._tool_stack = QStackedWidget()
        self._tool_panels: dict[str, QWidget] = {}

        layout = QVBoxLayout(self)
        layout.addWidget(self._video_container, stretch=1)
        layout.addLayout(controls_row)
        layout.addLayout(button_row)
        layout.addWidget(self._tool_stack)

    @property
    def player(self) -> QMediaPlayer:
        return self._player

    def register_panel(self, name: str, widget: QWidget) -> None:
        self._tool_panels[name] = widget
        self._tool_stack.addWidget(widget)

    def show_tool(self, name: str) -> None:
        widget = self._tool_panels.get(name)
        if widget is not None:
            self._tool_stack.setCurrentWidget(widget)
        button = self._tool_buttons.get(name)
        if button is not None:
            button.setChecked(True)

    def set_trim_range(self, start_sec: float, end_sec: float) -> None:
        self._trim_range_bar.set_range(start_sec * 1000, end_sec * 1000)

    def load_video(self, path: Path) -> None:
        self._player.stop()
        self._detection_overlay.clear()
        self._detection_frame_index = 0
        self._detection_busy = False
        self._resume_after_detection = False
        self._player.setSource(QUrl.fromLocalFile(str(path)))

    def _toggle_playback(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        is_playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._play_button.setText("Pause" if is_playing else "Play")
        self._play_button.setIcon(icons.pause_icon() if is_playing else icons.play_icon())

    def _on_slider_released(self) -> None:
        self._player.setPosition(self._position_slider.value())

    def _on_position_changed(self, position: int) -> None:
        # Don't fight the user's click/drag: while they're holding the handle
        # down, leave the slider's value alone until they release it.
        if self._position_slider.isSliderDown():
            return
        self._position_slider.blockSignals(True)
        self._position_slider.setValue(position)
        self._position_slider.blockSignals(False)
        self._time_label.setText(f"{_format_ms(position)} / {_format_ms(self._player.duration())}")

    def _on_duration_changed(self, duration: int) -> None:
        self._position_slider.setRange(0, duration)
        self._trim_range_bar.set_duration(duration)

    def _on_detect_toggled(self, checked: bool) -> None:
        if checked:
            self._video_stack.setCurrentWidget(self._detection_overlay)
            self._ensure_detection_model_loaded()
            self._connect_detection_sink()
        else:
            self._disconnect_detection_sink()
            self._video_stack.setCurrentWidget(self._video_widget)
            self._detection_overlay.clear()
            # Don't leave playback stuck paused if detection is switched off
            # while a frame is blocked waiting on an in-flight detection.
            if self._resume_after_detection:
                self._resume_after_detection = False
                self._player.play()

    def _ensure_detection_model_loaded(self) -> None:
        if self._detection_model is not None or self._detection_model_load_attempted:
            return
        self._detection_model_load_attempted = True
        model_path = find_onnx_model(DEFAULT_MODELS_DIR)
        if model_path is None:
            logger.warning(f"No .onnx model found in {DEFAULT_MODELS_DIR}; detection disabled")
            self._disable_detection(f"No .onnx model found in {DEFAULT_MODELS_DIR}")
            return
        worker = Worker(DetectionModel, model_path)
        # Bound-method connection so Qt marshals the result back onto the GUI
        # thread instead of running the callback inline on the worker thread.
        worker.signals.result.connect(self._on_detection_model_loaded)
        worker.signals.error.connect(self._on_detection_model_load_error)
        QThreadPool.globalInstance().start(worker)

    def _on_detection_model_loaded(self, model: object) -> None:
        self._detection_model = model  # type: ignore[assignment]

    def _on_detection_model_load_error(self, message: str) -> None:
        logger.warning(f"Failed to load detection model: {message}")
        self._disable_detection(f"Failed to load detection model: {message}")

    def _disable_detection(self, reason: str) -> None:
        self._detect_button.setChecked(False)
        self._detect_button.setEnabled(False)
        self._detect_button.setToolTip(reason)

    def _connect_detection_sink(self) -> None:
        if self._detection_sink_connected:
            return
        sink = self._player.videoSink()
        if sink is None:
            return
        sink.videoFrameChanged.connect(self._on_video_frame_changed)
        self._detection_sink_connected = True

    def _disconnect_detection_sink(self) -> None:
        if not self._detection_sink_connected:
            return
        sink = self._player.videoSink()
        if sink is not None:
            sink.videoFrameChanged.disconnect(self._on_video_frame_changed)
        self._detection_sink_connected = False

    def _on_video_frame_changed(self, frame: QVideoFrame) -> None:
        if self._detection_model is None or self._detection_busy or not frame.isValid():
            return
        rgb = frame_to_rgb_array(frame)
        if rgb is None:
            return
        # Block playback on this frame until the detection result for it comes
        # back - otherwise the player keeps decoding/presenting frames faster
        # than ONNX Runtime can process them and detection falls further and
        # further behind what's on screen. _on_detection_result/_on_detection_error
        # resume playback once this frame's detection is done.
        self._resume_after_detection = (
            self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        )
        self._player.pause()
        self._detection_busy = True
        self._detection_frame_index += 1
        worker = Worker(self._run_detection, rgb, self._detection_frame_index)
        worker.signals.result.connect(self._on_detection_result)
        worker.signals.error.connect(self._on_detection_error)
        QThreadPool.globalInstance().start(worker)

    def _run_detection(self, frame_rgb, frame_index: int) -> QImage:
        assert self._detection_model is not None
        detections = self._detection_model.infer(frame_rgb)
        logger.info(f"Detection frame {frame_index}: {len(detections)} detection(s)")
        annotated = annotate_frame(frame_rgb, detections)
        return rgb_array_to_qimage(annotated)

    def _on_detection_result(self, image: QImage) -> None:
        self._detection_busy = False
        if self._detect_button.isChecked():
            pixmap = QPixmap.fromImage(image)
            self._detection_overlay.setPixmap(
                pixmap.scaled(
                    self._detection_overlay.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self._resume_playback_if_pending()

    def _on_detection_error(self, message: str) -> None:
        self._detection_busy = False
        logger.warning(f"Detection failed: {message}")
        self._resume_playback_if_pending()

    def _resume_playback_if_pending(self) -> None:
        if self._resume_after_detection:
            self._resume_after_detection = False
            self._player.play()
