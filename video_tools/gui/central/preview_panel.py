from __future__ import annotations

from collections import deque
from time import perf_counter
from dataclasses import replace
from pathlib import Path

from loguru import logger
from PySide6.QtCore import QThreadPool, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoFrame
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from video_tools.config.settings import Settings, save_settings
from video_tools.core.jobs import Worker
from video_tools.core.image_sequence import ImageSequence, is_image_sequence
from video_tools.core.naming import unique_suffixed_output_path
from video_tools.detection import (
    DetectionConfig,
    DetectionModel,
    DetectionSaveJob,
    DetectionTrackingPipeline,
    annotate_frame,
    frame_to_rgb_array,
    rgb_array_to_qimage,
)
from video_tools.detection.model import list_detection_models, resolve_model_path
from video_tools.detection.timing import InferenceTiming
from video_tools.detection.mot import MotDetection, annotate_mot_frame, load_mot_detections
from video_tools.gui import icons
from video_tools.gui.central.detection_settings_panel import DetectionSettingsPanel
from video_tools.gui.widgets.progress_widget import ProgressWidget
from video_tools.gui.widgets.playhead_slider import PlayheadSlider
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


class _PreviewOverlay(QLabel):
    frame_painted = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self._frame_pending = False

    def setPixmap(self, pixmap: QPixmap) -> None:
        self._frame_pending = True
        super().setPixmap(pixmap)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._frame_pending:
            self._frame_pending = False
            self.frame_painted.emit(perf_counter())


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
        self._detection_overlay = _PreviewOverlay()
        self._detection_overlay.frame_painted.connect(self._on_preview_frame_painted)
        self._detection_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_container = QWidget()
        self._video_stack = QStackedLayout(self._video_container)
        self._video_stack.addWidget(self._video_widget)
        self._video_stack.addWidget(self._detection_overlay)
        self._video_stack.setCurrentWidget(self._video_widget)

        self._current_path: Path | None = None
        self._sequence: ImageSequence | None = None
        self._sequence_index = 0
        self._mot_detections: dict[int, list[MotDetection]] = {}
        self._sequence_timer = QTimer(self)
        self._sequence_timer.setSingleShot(True)
        self._sequence_timer.timeout.connect(self._advance_sequence)
        self._pipeline: DetectionTrackingPipeline | None = None
        self._detection_model: DetectionModel | None = None
        self._detection_config: DetectionConfig = settings.detection
        self._detection_model_load_attempted = False
        # Offline "Detect & Save" job (writes an annotated copy to disk).
        self._save_job: DetectionSaveJob | None = None
        # Set when Detect & Save is clicked before the model has finished loading;
        # the save starts as soon as the model arrives.
        self._pending_save = False
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

        self._preview_frame_times: deque[float] = deque(maxlen=1000)
        self._inference_timing = InferenceTiming()
        self._timing_generation = 0
        self._inference_label = QLabel("ONNX: —")
        self._inference_label.setToolTip(
            "Mean model execution time, excluding preprocessing and tracking. "
            "TensorRT and ONNX CUDA I/O binding exclude input/output transfers. "
            "Preview FPS measures painted frames over the last 2 seconds, including all delays. "
            "Skips 5 warm-up inferences; filters the last 300 using 1.5 × IQR."
        )
        self._detect_button = QPushButton("Detect")
        self._detect_button.setIcon(icons.detect_icon())
        self._detect_button.setCheckable(True)
        self._detect_button.toggled.connect(self._on_detect_toggled)

        self._model_combo = QComboBox()
        self._model_combo.setMinimumContentsLength(18)
        self._populate_model_combo()
        self._model_combo.currentIndexChanged.connect(self._on_main_model_changed)

        self._confidence_label = QLabel()
        self._confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self._confidence_slider.setRange(1, 99)
        self._confidence_slider.setSingleStep(1)
        self._confidence_slider.setPageStep(5)
        self._confidence_slider.setFixedWidth(120)
        self._confidence_slider.setToolTip("Detector confidence threshold")
        self._confidence_slider.setValue(round(self._detection_config.confidence * 100))
        self._confidence_slider.valueChanged.connect(
            self._on_confidence_slider_changed
        )
        self._confidence_commit_timer = QTimer(self)
        self._confidence_commit_timer.setSingleShot(True)
        self._confidence_commit_timer.setInterval(150)
        self._confidence_commit_timer.timeout.connect(self._commit_confidence_slider)
        self._update_confidence_label()

        self._mot_button = QPushButton("MOT det")
        self._mot_button.setCheckable(True)
        self._mot_button.setVisible(False)
        self._mot_button.setToolTip("Overlay detections from det/det.txt")
        self._mot_button.toggled.connect(self._on_mot_toggled)

        # "&&" renders as a single literal "&" (a single "&" would be a mnemonic).
        self._detect_save_button = QPushButton("Detect && Save")
        self._detect_save_button.setIcon(icons.save_icon())
        self._detect_save_button.setToolTip(
            "Run detection over the whole video and save an annotated copy (no preview)"
        )
        self._detect_save_button.clicked.connect(self._on_detect_save_clicked)

        self._detect_settings_button = QPushButton()
        self._detect_settings_button.setIcon(icons.settings_icon())
        self._detect_settings_button.setToolTip("Detection settings")
        self._detect_settings_button.clicked.connect(self._show_detection_settings)
        self._detection_settings_dialog: QDialog | None = None
        self._detection_settings_panel: DetectionSettingsPanel | None = None

        self._position_slider = PlayheadSlider()
        self._position_slider.setToolTip("Drag the playhead to seek")
        self._position_slider.seek_requested.connect(self._on_slider_moved)
        self._position_slider.seek_finished.connect(self._on_slider_released)

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

        timeline_row = QHBoxLayout()
        timeline_row.addLayout(slider_column, stretch=1)
        timeline_row.addWidget(self._time_label)

        controls_row = QHBoxLayout()
        controls_row.addWidget(self._play_button)
        controls_row.addWidget(self._detect_button)
        controls_row.addWidget(self._mot_button)
        controls_row.addWidget(QLabel("Model:"))
        controls_row.addWidget(self._model_combo)
        controls_row.addWidget(self._confidence_label)
        controls_row.addWidget(self._confidence_slider)
        controls_row.addWidget(self._detect_save_button)
        controls_row.addWidget(self._detect_settings_button)
        controls_row.addStretch(1)

        # Progress row for the offline Detect & Save job; hidden until one runs.
        self._save_progress = ProgressWidget()

        button_row = QHBoxLayout()

        info_button = QPushButton("Info")
        info_button.setIcon(icons.info_icon())
        info_button.clicked.connect(self.info_clicked)
        button_row.addWidget(info_button)

        self._tool_buttons: dict[str, QPushButton] = {}
        for name in TOOL_NAMES:
            button = QPushButton(name)
            button.setIcon(TOOL_ICON_FUNCS[name]())
            button.clicked.connect(lambda checked=False, n=name: self.show_tool(n))
            button_row.addWidget(button)
            self._tool_buttons[name] = button
        button_row.addStretch(1)

        self._inference_panel = QGroupBox("Inference statistics")
        inference_layout = QVBoxLayout(self._inference_panel)
        self._inference_label.setWordWrap(True)
        inference_layout.addWidget(self._inference_label)
        self._preview_fps_label = QLabel("Preview FPS: 0.0")
        inference_layout.addWidget(self._preview_fps_label)
        self._fps_timer = QTimer(self)
        self._fps_timer.setInterval(250)
        self._fps_timer.timeout.connect(self._update_preview_fps)
        self._fps_timer.start()

        # Each tool's parameters live in a pop-up dialog (built lazily on first
        # open) rather than an inline panel, keeping the main window compact.
        self._tool_panels: dict[str, QWidget] = {}
        self._tool_dialogs: dict[str, QDialog] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self._video_container, stretch=1)
        layout.addLayout(timeline_row)
        layout.addLayout(controls_row)
        layout.addWidget(self._save_progress)
        layout.addLayout(button_row)
        layout.addWidget(self._inference_panel)

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
        self._reset_inference_timing()
        self._current_path = path
        self._sequence_timer.stop()
        self._sequence = None
        self._mot_detections = {}
        self._mot_button.setChecked(False)
        self._mot_button.setVisible(False)
        self._disconnect_detection_sink()
        self._player.stop()
        self._intended_playing = False
        self._update_play_button()
        # Leave a running save's progress row alone; only clear it when idle.
        if self._save_job is None:
            self._save_progress.reset()
        self._detection_overlay.clear()
        self._detection_frame_index = 0
        self._detection_busy = False
        self._resume_after_detection = False
        # Trackers hold onto the previous clip's boxes; drop them for the new video.
        if self._pipeline is not None:
            self._pipeline.reset()
        sequence_input = is_image_sequence(path)
        for button in self._tool_buttons.values():
            button.setEnabled(not sequence_input)
            button.setToolTip("Not available for image sequences" if sequence_input else "")
        if sequence_input:
            try:
                self._sequence = ImageSequence(
                    path,
                    cache_size=self._settings.image_cache_size,
                    default_fps=self._settings.image_sequence_fps,
                )
                det_path = self._sequence.detection_file
                if det_path is not None:
                    self._mot_detections = load_mot_detections(det_path)
                    self._mot_button.setVisible(True)
                    self._mot_button.setToolTip(f"Overlay {det_path}")
                self._sequence_index = 0
                self._on_duration_changed(self._sequence.duration_ms)
                self._video_stack.setCurrentWidget(self._detection_overlay)
                self._display_sequence_frame()
            except (OSError, RuntimeError, ValueError) as exc:
                QMessageBox.warning(self, "Could not read image sequence", str(exc))
            return
        self._player.setSource(QUrl.fromLocalFile(str(path)))
        if self._detect_button.isChecked():
            self._video_stack.setCurrentWidget(self._detection_overlay)
            self._connect_detection_sink()
        else:
            self._video_stack.setCurrentWidget(self._video_widget)

    def _toggle_playback(self) -> None:
        self._set_playing(not self._intended_playing)

    def _set_playing(self, playing: bool) -> None:
        if playing != self._intended_playing:
            self._preview_frame_times.clear()
        self._intended_playing = playing
        if self._sequence is not None:
            if playing:
                if self._sequence_index >= len(self._sequence) - 1:
                    self._sequence_index = 0
                    self._display_sequence_frame()
                self._schedule_sequence_frame()
            else:
                self._sequence_timer.stop()
        elif playing:
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
        if self._sequence is not None:
            index = round(self._position_slider.value() * self._sequence.fps / 1000)
            self._sequence_index = min(len(self._sequence) - 1, max(0, index))
            self._display_sequence_frame()
        else:
            self._player.setPosition(self._position_slider.value())

    def _on_slider_moved(self, position: int) -> None:
        self._update_time_label(position)
        if self._sequence is not None:
            index = round(position * self._sequence.fps / 1000)
            index = min(len(self._sequence) - 1, max(0, index))
            if index != self._sequence_index:
                self._sequence_index = index
                self._display_sequence_frame()
        else:
            self._player.setPosition(position)

    def _on_position_changed(self, position: int) -> None:
        # Don't fight the user's click/drag: while they're holding the handle
        # down, leave the slider's value alone until they release it.
        if self._position_slider.isSliderDown():
            return
        self._position_slider.blockSignals(True)
        self._position_slider.setValue(position)
        self._position_slider.blockSignals(False)
        self._update_time_label(position)

    def _update_time_label(self, position: int) -> None:
        duration = (
            self._sequence.duration_ms
            if self._sequence is not None
            else self._player.duration()
        )
        self._time_label.setText(f"{_format_ms(position)} / {_format_ms(duration)}")

    def _on_duration_changed(self, duration: int) -> None:
        self._position_slider.setRange(0, duration)
        self._trim_range_bar.set_duration(duration)

    def _on_preview_frame_painted(self, timestamp: float) -> None:
        if self._intended_playing:
            self._preview_frame_times.append(timestamp)

    def _update_preview_fps(self) -> None:
        now = perf_counter()
        while self._preview_frame_times and self._preview_frame_times[0] < now - 2:
            self._preview_frame_times.popleft()
        fps = 0.0
        if self._intended_playing and len(self._preview_frame_times) >= 2:
            elapsed = now - self._preview_frame_times[0]
            if elapsed > 0:
                fps = (len(self._preview_frame_times) - 1) / elapsed
        self._preview_fps_label.setText(f"Preview FPS: {fps:.1f}")

    def _reset_inference_timing(self) -> None:
        self._preview_frame_times.clear()
        self._preview_fps_label.setText("Preview FPS: 0.0")
        self._timing_generation += 1
        backend = self._detection_model.backend_name if self._detection_model else "Model"
        self._inference_timing = InferenceTiming(backend)
        self._inference_label.setText(f"{backend}: —")

    def _on_detect_toggled(self, checked: bool) -> None:
        self._reset_inference_timing()
        if checked:
            self._video_stack.setCurrentWidget(self._detection_overlay)
            self._ensure_detection_model_loaded()
            if self._sequence is not None:
                self._display_sequence_frame()
            else:
                self._connect_detection_sink()
        else:
            if self._sequence is None:
                self._disconnect_detection_sink()
                self._video_stack.setCurrentWidget(self._video_widget)
                self._detection_overlay.clear()
            else:
                self._display_sequence_frame()
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
        model_path = resolve_model_path(filename, self._settings.models_dir)
        if model_path is None:
            reason = f"No ONNX or TensorRT model found in {self._settings.models_dir}"
            logger.warning(f"{reason}; detection disabled")
            self._disable_detection(reason)
            return
        self._pending_model_filename = model_path.name
        worker = Worker(DetectionModel, model_path)
        # Bound-method connection so Qt marshals the result back onto the GUI
        # thread instead of running the callback inline on the worker thread.
        worker.signals.result.connect(self._on_detection_model_loaded)
        worker.signals.error.connect(self._on_detection_model_load_error)
        QThreadPool.globalInstance().start(worker)

    def _populate_model_combo(self) -> None:
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        models = list_detection_models(self._settings.models_dir)
        for name in models:
            self._model_combo.addItem(name, userData=name)
        resolved = resolve_model_path(
            self._detection_config.model_filename, self._settings.models_dir
        )
        if resolved is None:
            self._model_combo.addItem("(no ONNX models)", userData=None)
            self._model_combo.setEnabled(False)
            self._model_combo.setToolTip(
                f"No ONNX or TensorRT models found in {self._settings.models_dir}"
            )
        else:
            index = self._model_combo.findData(resolved.name)
            self._model_combo.setCurrentIndex(max(0, index))
            self._model_combo.setEnabled(True)
            self._model_combo.setToolTip(str(resolved))
        self._model_combo.blockSignals(False)

    def _on_main_model_changed(self, _index: int) -> None:
        filename = self._model_combo.currentData()
        if filename is None:
            return
        self._model_combo.setToolTip(str(self._settings.models_dir / filename))
        self._on_detection_config_changed(
            replace(self._detection_config, model_filename=filename)
        )

    def _on_confidence_slider_changed(self, _value: int) -> None:
        self._update_confidence_label()
        self._confidence_commit_timer.start()

    def _update_confidence_label(self) -> None:
        confidence = self._confidence_slider.value() / 100
        self._confidence_label.setText(f"Confidence: {confidence:.2f}")

    def _commit_confidence_slider(self) -> None:
        confidence = self._confidence_slider.value() / 100
        if confidence == self._detection_config.confidence:
            return
        self._on_detection_config_changed(
            replace(self._detection_config, confidence=confidence)
        )

    def _on_detection_model_loaded(self, model: object) -> None:
        detection_model: DetectionModel = model  # type: ignore[assignment]
        self._detection_model = detection_model
        self._detect_button.setEnabled(True)
        self._detect_button.setToolTip("")
        self._detect_save_button.setEnabled(True)
        self._detect_save_button.setToolTip(
            "Run detection over the whole video and save an annotated copy (no preview)"
        )
        self._reset_inference_timing()
        if self._pipeline is None:
            self._pipeline = DetectionTrackingPipeline(
                detection_model,
                self._detection_config,
                log_frames=self._settings.log_frames,
            )
        else:
            self._pipeline.set_model(detection_model)
            self._pipeline.update_config(self._detection_config)
        self._loaded_model_filename = self._pending_model_filename
        if self._sequence is not None and self._detect_button.isChecked():
            self._display_sequence_frame()
        # A Detect & Save click that arrived before the model was ready starts now.
        if self._pending_save:
            self._pending_save = False
            self._start_detection_save()

    def _on_detection_model_load_error(self, message: str) -> None:
        logger.warning(f"Failed to load detection model: {message}")
        self._disable_detection(f"Failed to load detection model: {message}")

    def _disable_detection(self, reason: str) -> None:
        self._detect_button.setChecked(False)
        self._detect_button.setEnabled(False)
        self._detect_button.setToolTip(reason)
        self._detect_save_button.setEnabled(False)
        self._detect_save_button.setToolTip(reason)
        # Don't leave a queued Detect & Save waiting on a model that won't load.
        if self._pending_save:
            self._pending_save = False
            QMessageBox.warning(self, "Detection unavailable", reason)

    def _show_detection_settings(self) -> None:
        if self._detection_settings_dialog is None:
            panel = DetectionSettingsPanel(
                self._detection_config, self._settings.models_dir
            )
            panel.config_changed.connect(self._on_detection_config_changed)
            dialog = QDialog(self)
            dialog.setWindowTitle("Detection settings")
            layout = QVBoxLayout(dialog)
            layout.addWidget(panel)
            self._detection_settings_dialog = dialog
            self._detection_settings_panel = panel
        self._detection_settings_dialog.show()
        self._detection_settings_dialog.raise_()
        self._detection_settings_dialog.activateWindow()

    def _on_detection_config_changed(self, config: DetectionConfig) -> None:
        self._reset_inference_timing()
        self._confidence_commit_timer.stop()
        model_changed = config.model_filename != self._detection_config.model_filename
        self._detection_config = config
        confidence_value = round(config.confidence * 100)
        if confidence_value != self._confidence_slider.value():
            self._confidence_slider.blockSignals(True)
            self._confidence_slider.setValue(confidence_value)
            self._confidence_slider.blockSignals(False)
            self._update_confidence_label()
        model_index = self._model_combo.findData(config.model_filename)
        if model_index >= 0 and model_index != self._model_combo.currentIndex():
            self._model_combo.blockSignals(True)
            self._model_combo.setCurrentIndex(model_index)
            self._model_combo.blockSignals(False)
            self._model_combo.setToolTip(
                str(self._settings.models_dir / config.model_filename)
            )
        if self._detection_settings_panel is not None:
            self._detection_settings_panel.set_config(config)
        self._settings.detection = config
        save_settings(self._settings)
        if self._pipeline is None:
            if model_changed and self._detection_model_load_attempted:
                self._load_model(config.model_filename)
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
        worker = Worker(self._run_detection, rgb, self._detection_frame_index, self._timing_generation)
        worker.signals.result.connect(self._on_detection_result)
        worker.signals.error.connect(self._on_detection_error)
        QThreadPool.globalInstance().start(worker)

    def _run_detection(
        self, frame_rgb, frame_index: int, generation: int
    ) -> tuple[QImage, float | None, int]:
        assert self._pipeline is not None
        detections = self._pipeline.process(frame_rgb)
        inference_ms = self._pipeline.last_inference_ms
        annotated = annotate_frame(frame_rgb, detections)
        if self._sequence is not None and self._mot_button.isChecked():
            annotated = annotate_mot_frame(
                annotated, self._mot_detections.get(frame_index + 1, [])
            )
        return rgb_array_to_qimage(annotated), inference_ms, generation

    def _on_detection_result(self, result: tuple[QImage, float | None, int]) -> None:
        image, inference_ms, generation = result
        if self._detect_button.isChecked() and generation == self._timing_generation:
            if inference_ms is not None:
                self._inference_timing.add(inference_ms)
                self._inference_label.setText(self._inference_timing.label())
        self._detection_busy = False
        if self._detect_button.isChecked():
            self._set_overlay_image(image)
        elif self._sequence is not None:
            self._display_sequence_frame()
        self._resume_playback_if_pending()
        if self._sequence is not None and self._intended_playing:
            self._schedule_sequence_frame()

    def _on_detection_error(self, message: str) -> None:
        self._detection_busy = False
        logger.warning(f"Detection failed: {message}")
        self._resume_playback_if_pending()
        if self._sequence is not None and self._intended_playing:
            self._schedule_sequence_frame()

    def _resume_playback_if_pending(self) -> None:
        if self._resume_after_detection:
            self._resume_after_detection = False
            self._player.play()

    # --- Numbered image sequences -----------------------------------------

    def _on_mot_toggled(self, _checked: bool) -> None:
        if self._sequence is not None:
            self._display_sequence_frame()

    def _schedule_sequence_frame(self) -> None:
        if (
            self._sequence is not None
            and self._intended_playing
            and not self._detection_busy
        ):
            self._sequence_timer.start(self._sequence.frame_interval_ms)

    def _advance_sequence(self) -> None:
        if self._sequence is None or not self._intended_playing:
            return
        if self._sequence_index >= len(self._sequence) - 1:
            self._set_playing(False)
            return
        self._sequence_index += 1
        self._display_sequence_frame()

    def _display_sequence_frame(self) -> None:
        sequence = self._sequence
        if sequence is None or self._detection_busy:
            return
        try:
            frame_rgb = sequence.frame(self._sequence_index)
        except (OSError, RuntimeError, IndexError) as exc:
            self._set_playing(False)
            QMessageBox.warning(self, "Could not read frame", str(exc))
            return

        position = round(self._sequence_index / sequence.fps * 1000)
        self._on_position_changed(position)
        if self._detect_button.isChecked() and self._pipeline is not None:
            self._apply_pending_config()
            self._sequence_timer.stop()
            self._detection_busy = True
            worker = Worker(self._run_detection, frame_rgb, self._sequence_index, self._timing_generation)
            worker.signals.result.connect(self._on_detection_result)
            worker.signals.error.connect(self._on_detection_error)
            QThreadPool.globalInstance().start(worker)
            return

        annotated = frame_rgb
        if self._mot_button.isChecked():
            annotated = annotate_mot_frame(
                frame_rgb, self._mot_detections.get(self._sequence_index + 1, [])
            )
        self._set_overlay_image(rgb_array_to_qimage(annotated))
        worker = Worker(sequence.prefetch, self._sequence_index + 1, 4)
        QThreadPool.globalInstance().start(worker)
        self._schedule_sequence_frame()

    def _set_overlay_image(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        self._detection_overlay.setPixmap(
            pixmap.scaled(
                self._detection_overlay.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    # --- Detect & Save (offline, no preview) --------------------------------

    def _on_detect_save_clicked(self) -> None:
        if self._current_path is None or self._save_job is not None:
            return
        # The offline save and live detection share one ONNX session/annotators;
        # turn live detection off so they never run at the same time.
        if self._detect_button.isChecked():
            self._detect_button.setChecked(False)
        self._ensure_detection_model_loaded()
        if self._detection_model is None:
            # Model still loading (or unavailable); start once it arrives, unless
            # loading fails - _disable_detection clears this flag in that case.
            self._pending_save = True
            return
        self._start_detection_save()

    def _start_detection_save(self) -> None:
        if self._current_path is None or self._detection_model is None:
            return
        dest = unique_suffixed_output_path(
            self._current_path, "detected", self._settings.output_dir, ext="mp4"
        )
        job = DetectionSaveJob(
            self._current_path,
            dest,
            self._detection_model,
            self._detection_config,
            image_sequence_fps=self._settings.image_sequence_fps,
            image_cache_size=self._settings.image_cache_size,
            log_frames=self._settings.log_frames,
            parent=self,
        )
        self._save_job = job
        self._detect_save_button.setEnabled(False)
        self._detect_button.setEnabled(False)
        self._save_progress.bind(job)
        job.finished.connect(self._on_save_finished)
        job.start()

    def _on_save_finished(self, success: bool, message: str) -> None:
        self._save_job = None
        self._detect_save_button.setEnabled(True)
        self._detect_button.setEnabled(True)
        if success:
            QMessageBox.information(self, "Detection saved", message)
        else:
            QMessageBox.warning(self, "Detection save failed", message)
