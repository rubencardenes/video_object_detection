from __future__ import annotations

from pathlib import Path

from loguru import logger
from PySide6.QtCore import QThreadPool, QUrl, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoFrame
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from video_tools.config.settings import Settings, save_settings
from video_tools.core.jobs import Worker
from video_tools.detection import (
    DEFAULT_MODELS_DIR,
    DetectionConfig,
    DetectionModel,
    DetectionTrackingPipeline,
    annotate_frame,
    frame_to_rgb_array,
    rgb_array_to_qimage,
)
from video_tools.detection.model import resolve_model_path
from video_tools.gui import icons
from video_tools.gui.central.detection_settings_panel import DetectionSettingsPanel
from video_tools.gui.widgets.trim_range_bar import TrimRangeBar

# Tool buttons in the bottom row. Each opens its parameters in a pop-up dialog
# (show_tool); Info has the same behaviour but its dialog is owned by MainWindow.
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

    def __init__(self, settings: Settings, parent: QWidget | None = None):
        super().__init__(parent)

        self._settings = settings
        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_output)
        self._video_widget = QVideoWidget()
        self._video_widget.setMinimumHeight(200)
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

        self._pipeline: DetectionTrackingPipeline | None = None
        self._detection_config: DetectionConfig = settings.detection
        self._detection_model_load_attempted = False
        self._detection_busy = False
        self._detection_sink_connected = False
        self._resume_after_detection = False
        self._detection_frame_index = 0
        # Filename the pipeline's model was loaded from, to detect model changes.
        self._loaded_model_filename: str | None = None
        self._pending_model_filename: str | None = None
        # Config edited while a detection worker is in flight; applied between frames.
        self._pending_config: DetectionConfig | None = None
        # The playback state the user asked for. Detection throttles playback by
        # pausing/resuming the player around every frame, which would otherwise
        # make the Play/Pause button flicker; the button tracks this intent
        # instead of the transient player state.
        self._intended_playing = False

        self._play_button = QPushButton("Play")
        self._play_button.setIcon(icons.play_icon())
        self._play_button.clicked.connect(self._toggle_playback)

        self._detect_button = QPushButton("Detect")
        self._detect_button.setIcon(icons.detect_icon())
        self._detect_button.setCheckable(True)
        self._detect_button.toggled.connect(self._on_detect_toggled)

        self._detect_settings_button = QPushButton()
        self._detect_settings_button.setIcon(icons.settings_icon())
        self._detect_settings_button.setToolTip("Detection settings")
        self._detect_settings_button.clicked.connect(self._show_detection_settings)
        self._detection_settings_dialog: QDialog | None = None

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
        self._player.mediaStatusChanged.connect(self._on_media_status_changed)

        slider_column = QVBoxLayout()
        slider_column.setSpacing(2)
        slider_column.addWidget(self._position_slider)
        slider_column.addWidget(self._trim_range_bar)

        controls_row = QHBoxLayout()
        controls_row.addWidget(self._play_button)
        controls_row.addWidget(self._detect_button)
        controls_row.addWidget(self._detect_settings_button)
        controls_row.addLayout(slider_column, stretch=1)
        controls_row.addWidget(self._time_label)

        button_row = QHBoxLayout()

        info_button = QPushButton("Info")
        info_button.setIcon(icons.info_icon())
        info_button.clicked.connect(self.info_clicked)
        button_row.addWidget(info_button)

        for name in TOOL_NAMES:
            button = QPushButton(name)
            button.setIcon(TOOL_ICON_FUNCS[name]())
            button.clicked.connect(lambda checked=False, n=name: self.show_tool(n))
            button_row.addWidget(button)
        button_row.addStretch(1)

        # Each tool's parameters live in a pop-up dialog (built lazily on first
        # open) rather than an inline panel, keeping the main window compact.
        self._tool_panels: dict[str, QWidget] = {}
        self._tool_dialogs: dict[str, QDialog] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self._video_container, stretch=1)
        layout.addLayout(controls_row)
        layout.addLayout(button_row)

    @property
    def player(self) -> QMediaPlayer:
        return self._player

    def register_panel(self, name: str, widget: QWidget) -> None:
        self._tool_panels[name] = widget

    def show_tool(self, name: str) -> None:
        widget = self._tool_panels.get(name)
        if widget is None:
            return
        dialog = self._tool_dialogs.get(name)
        if dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle(name)
            dialog_layout = QVBoxLayout(dialog)
            dialog_layout.addWidget(widget)
            self._tool_dialogs[name] = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def set_trim_range(self, start_sec: float, end_sec: float) -> None:
        self._trim_range_bar.set_range(start_sec * 1000, end_sec * 1000)

    def load_video(self, path: Path) -> None:
        self._player.stop()
        self._intended_playing = False
        self._update_play_button()
        self._detection_overlay.clear()
        self._detection_frame_index = 0
        self._detection_busy = False
        self._resume_after_detection = False
        # Trackers hold onto the previous clip's boxes; drop them for the new video.
        if self._pipeline is not None:
            self._pipeline.reset()
        self._player.setSource(QUrl.fromLocalFile(str(path)))

    def _toggle_playback(self) -> None:
        self._set_playing(not self._intended_playing)

    def _set_playing(self, playing: bool) -> None:
        self._intended_playing = playing
        if playing:
            self._player.play()
        else:
            # A pause requested by the user must also cancel any pending auto-resume
            # from an in-flight detection, or the throttle would restart playback.
            self._resume_after_detection = False
            self._player.pause()
        self._update_play_button()

    def _update_play_button(self) -> None:
        self._play_button.setText("Pause" if self._intended_playing else "Play")
        self._play_button.setIcon(
            icons.pause_icon() if self._intended_playing else icons.play_icon()
        )

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        # Only the Stopped state (natural end of media, or an explicit stop())
        # updates the button. Playing/Paused transitions are ignored because
        # detection toggles them many times a second to throttle the ONNX loop;
        # reacting to those is exactly what made the button flicker.
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self._intended_playing = False
            self._update_play_button()

    def _on_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        # When the clip ends, drop back to the Play state. Handled here (not only
        # via StoppedState) because during detection the player is often sitting
        # in PausedState from the per-frame throttle when the end is reached.
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self._resume_after_detection = False
            self._set_playing(False)

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
        if self._pipeline is not None or self._detection_model_load_attempted:
            return
        self._detection_model_load_attempted = True
        self._load_model(self._detection_config.model_filename)

    def _load_model(self, filename: str | None) -> None:
        model_path = resolve_model_path(filename, DEFAULT_MODELS_DIR)
        if model_path is None:
            logger.warning(f"No .onnx model found in {DEFAULT_MODELS_DIR}; detection disabled")
            self._disable_detection(f"No .onnx model found in {DEFAULT_MODELS_DIR}")
            return
        self._pending_model_filename = model_path.name
        worker = Worker(DetectionModel, model_path)
        # Bound-method connection so Qt marshals the result back onto the GUI
        # thread instead of running the callback inline on the worker thread.
        worker.signals.result.connect(self._on_detection_model_loaded)
        worker.signals.error.connect(self._on_detection_model_load_error)
        QThreadPool.globalInstance().start(worker)

    def _on_detection_model_loaded(self, model: object) -> None:
        detection_model: DetectionModel = model  # type: ignore[assignment]
        if self._pipeline is None:
            self._pipeline = DetectionTrackingPipeline(detection_model, self._detection_config)
        else:
            self._pipeline.set_model(detection_model)
            self._pipeline.update_config(self._detection_config)
        self._loaded_model_filename = self._pending_model_filename

    def _on_detection_model_load_error(self, message: str) -> None:
        logger.warning(f"Failed to load detection model: {message}")
        self._disable_detection(f"Failed to load detection model: {message}")

    def _disable_detection(self, reason: str) -> None:
        self._detect_button.setChecked(False)
        self._detect_button.setEnabled(False)
        self._detect_button.setToolTip(reason)

    def _show_detection_settings(self) -> None:
        if self._detection_settings_dialog is None:
            panel = DetectionSettingsPanel(self._detection_config)
            panel.config_changed.connect(self._on_detection_config_changed)
            dialog = QDialog(self)
            dialog.setWindowTitle("Detection settings")
            layout = QVBoxLayout(dialog)
            layout.addWidget(panel)
            self._detection_settings_dialog = dialog
        self._detection_settings_dialog.show()
        self._detection_settings_dialog.raise_()
        self._detection_settings_dialog.activateWindow()

    def _on_detection_config_changed(self, config: DetectionConfig) -> None:
        model_changed = config.model_filename != self._detection_config.model_filename
        self._detection_config = config
        self._settings.detection = config
        save_settings(self._settings)
        if self._pipeline is None:
            return  # built with this config when detection is first switched on
        if model_changed:
            self._load_model(config.model_filename)  # reload; config applied on result
        elif self._detection_busy:
            self._pending_config = config  # applied between frames (see _on_video_frame_changed)
        else:
            self._pipeline.update_config(config)

    def _apply_pending_config(self) -> None:
        if self._pending_config is not None and self._pipeline is not None:
            self._pipeline.update_config(self._pending_config)
            self._pending_config = None

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
        if self._pipeline is None or self._detection_busy or not frame.isValid():
            return
        # Not busy here, so no worker is touching the pipeline: safe to apply a
        # config change queued while the previous frame was still processing.
        self._apply_pending_config()
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
        assert self._pipeline is not None
        detections = self._pipeline.process(frame_rgb)
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
