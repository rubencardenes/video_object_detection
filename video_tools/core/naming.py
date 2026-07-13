from __future__ import annotations

from pathlib import Path


def suffixed_output_path(
    source: Path, suffix: str, output_dir: Path | None = None, ext: str | None = None
) -> Path:
    directory = output_dir if output_dir is not None else source.parent
    extension = ext if ext is not None else source.suffix.lstrip(".")
    return directory / f"{source.stem}_{suffix}.{extension}"
