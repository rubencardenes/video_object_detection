# Video and MOT image-sequence input

In addition to normal video files, the sidebar discovers numbered image folders.
Both of these layouts are supported:

```text
frames/000001.jpg
frames/000002.jpg

MOT20-01/
  seqinfo.ini
  img1/000001.jpg
  det/det.txt
```

Frames are naturally sorted by their numeric filename. MOT sequences use
`frameRate` from `seqinfo.ini`; other folders use `image_sequence_fps` from
`settings.yaml`. Decoded images are kept in a bounded LRU cache controlled by
`image_cache_size`, with the following frames prefetched in the background.

When `det/det.txt` is present, the preview shows a **MOT det** toggle. It reads
standard MOTChallenge rows (`frame,id,left,top,width,height,confidence,...`) and
draws the boxes, IDs and confidence values for the current frame.

## Detection models

`models_dir` in `settings.yaml` controls where ONNX and TensorRT models are discovered; its
default is `~/data/models`. `detection.model_filename` is resolved relative to
that directory. The detector supports both RF-DETR exports (`dets` + `labels`
outputs) and Ultralytics YOLO detection exports such as `yolo26l.onnx`.
The active model is shown and can be changed from the **Model** selector in the
main preview controls. Set `log_frames: true` to enable per-frame Detect/Track
messages; they are disabled by default.

TensorRT `.engine` and `.plan` files can also be selected from **Model**. Install
the optional runtime with `uv sync --extra tensorrt` (requires an NVIDIA driver
and CUDA 13 runtime). Engines must be compatible
with the installed TensorRT version and NVIDIA GPU. Raw engine files and
Ultralytics engines with a metadata header are accepted. Inputs must be batch-one
NCHW RGB, FP32 or FP16; dynamic inputs use profile 0's optimal shape. The backend
uses reusable CUDA buffers and the same detection postprocessing as ONNX.

The statistics panel identifies the active backend. TensorRT execution timing
excludes input/output transfers, preprocessing, and postprocessing; **Preview FPS**
continues to measure displayed frames including those costs.
