from __future__ import annotations

from pathlib import Path


def suffixed_output_path(
    source: Path, suffix: str, output_dir: Path | None = None, ext: str | None = None
) -> Path:
    directory = output_dir if output_dir is not None else source.parent
    extension = ext if ext is not None else source.suffix.lstrip(".")
    return directory / f"{source.stem}_{suffix}.{extension}"


def unique_suffixed_output_path(
    source: Path, suffix: str, output_dir: Path | None = None, ext: str | None = None
) -> Path:
    """Like suffixed_output_path, but never returns a path that already exists.

    Appends _1, _2, ... to the suffix until a free path is found, so repeated
    runs produce foo_detected.mp4, foo_detected_1.mp4, ... instead of overwriting.
    """
    candidate = suffixed_output_path(source, suffix, output_dir, ext)
    counter = 1
    while candidate.exists():
        candidate = suffixed_output_path(source, f"{suffix}_{counter}", output_dir, ext)
        counter += 1
    return candidate
