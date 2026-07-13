# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A PySide6 desktop app for browsing and processing a local video library: recursively lists videos from a configured folder, previews them, and runs Convert / Cut / Resize / FPS operations, plus a metadata Info view. `ffmpeg`/`ffprobe` (external binaries, via subprocess/`QProcess`) do the actual video processing and metadata probing; `opencv-python` (`cv2`) is used only for thumbnail/frame extraction and as an metadata fallback when ffprobe is unavailable; Qt Multimedia (`QMediaPlayer`/`QVideoWidget`) drives in-app playback and scrubbing.

## Setup and running

This project uses `uv` (see `pyproject.toml`, `.python-version` pinned to 3.12). `ffmpeg`/`ffprobe` must be installed and on `PATH` (or pointed to via `ffmpeg_path`/`ffprobe_path` in `settings.yaml`) for processing and rich metadata; the app still runs without them, with processing actions disabled and Info falling back to `cv2`-derived fields.

```bash
uv sync                # install dependencies
uv run python main.py  # launch the app
```

There are no tests, lint config, or build steps configured in this repo.

## Configuration

`settings.yaml` at the repo root (gitignored — personal folder paths) configures the app; `settings.example.yaml` documents the schema and is the file to copy from. Loaded/saved via `video_tools/config/settings.py` (`Settings` dataclass, `load_settings`/`save_settings`). If `settings.yaml` is missing, malformed, or its `videos_root` doesn't exist, the app falls back to in-memory defaults rather than crashing (see `load_settings`) — on first run it also prompts once via a folder picker (`main.py`).

## Architecture

```
video_tools/
  config/settings.py     # Settings dataclass, YAML load/save, defaults/fallback
  core/                  # No Qt-widget code; pure logic + background job plumbing
    metadata.py            # probe_video(): ffprobe first, cv2 fallback -> VideoInfo
    thumbnails.py           # extract_thumbnail(): cv2 frame grab -> QImage
    ffmpeg_runner.py         # find_ffmpeg() + pure argv builders (build_convert_cmd, build_cut_cmd, build_resize_cmd, build_fps_cmd)
    jobs.py                   # Worker (QRunnable, generic background callable), ThumbnailWorker, FfmpegJob (QProcess wrapper with progress parsing)
    naming.py                  # suffixed_output_path(): e.g. foo.mp4 -> foo_converted.mp4
  gui/
    main_window.py          # QSplitter(central stack, sidebar); owns "currently selected video"
    sidebar.py                # recursive video list: filter box, refresh, change-folder, lazy thumbnails
    central/
      preview_panel.py         # QMediaPlayer + QVideoWidget + action button row (constant chrome)
      ffmpeg_task_panel.py       # base class for Convert/Resize/FPS/Cut: form + Run + progress wiring
      info_panel.py / convert_panel.py / resize_panel.py / fps_panel.py / cut_panel.py
      trim_scrubber.py           # in/out point spinboxes + "set from playhead" buttons, used by CutPanel
    widgets/progress_widget.py  # QProgressBar + status + cancel, binds to an FfmpegJob
main.py                    # entry point: loads settings, first-run folder picker, shows MainWindow
```

**Threading rule:** nothing on the GUI thread calls `subprocess`, `cv2`, or other blocking I/O directly. ffmpeg operations run through `FfmpegJob` (wraps `QProcess`, parses `-progress pipe:2` stderr output for percent-complete, emits `progress`/`finished`). One-shot probing/thumbnailing runs through `Worker`/`ThumbnailWorker` (`QRunnable` on `QThreadPool.globalInstance()`).

**Cross-thread signal gotcha:** connect `QRunnable`/`Worker` signals to *bound methods* of a `QObject`, never to a bare lambda or function. Qt determines a queued (thread-safe) connection by the receiver's `QObject` affinity; a plain lambda has none, so the callback silently runs synchronously on the worker thread instead of the GUI thread. `FfmpegTaskPanel._on_probe_result` is the pattern to follow — see the comment at its call site.

**FfmpegTaskPanel subclassing:** `ConvertPanel`/`ResizePanel`/`FpsPanel`/`CutPanel` all extend `FfmpegTaskPanel` (`gui/central/ffmpeg_task_panel.py`), implementing `output_suffix()` and `build_command(src, dst, ffmpeg)`, and optionally overriding `job_duration_sec(info)` (Cut needs the trim span, not the source's full duration, for correct progress-percent math). Convert/Resize/FPS re-encode to `libx264`, so their `build_command` forces the output extension to `.mp4` (a source `.webm`/`.mkv` container can't hold an h264/aac stream) and their ffmpeg builders apply a `scale=trunc(iw/2)*2:trunc(ih/2)*2` filter (libx264 requires even width/height; screen recordings and similar often aren't). Cut uses `-c copy` (stream copy, no re-encode) and keeps the source's own extension, but is therefore only accurate to the nearest keyframe.

**Sidebar thumbnails:** generated once per scan (`SidebarWidget._start_thumbnail_jobs`, cached in `_thumbnail_cache` by path) rather than regenerated on every filter keystroke; the list's display icon size is deliberately smaller than the extraction size so thumbnails don't crowd out the filename text.
