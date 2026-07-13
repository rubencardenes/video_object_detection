from __future__ import annotations

import cv2

# Single-object visual trackers that follow a box from image pixels alone (no
# detector call). CSRT is the most accurate, KCF faster, MOSSE fastest, MIL the
# widely-available fallback. Which of these exist depends on the OpenCV build
# (CSRT/KCF/MOSSE ship in opencv-contrib-python), so available_methods() probes.
TRACKING_METHODS = ("CSRT", "KCF", "MOSSE", "MIL")


def _constructor(method: str):
    """Return a zero-arg callable that builds a fresh cv2 tracker, or None if unavailable.

    OpenCV moved these constructors around across versions, so try the known
    spellings in order: cv2.TrackerX_create, cv2.TrackerX.create, and the
    cv2.legacy.* variants (some trackers only live in the legacy module).
    """
    for namespace in (cv2, getattr(cv2, "legacy", None)):
        if namespace is None:
            continue
        factory = getattr(namespace, f"Tracker{method}_create", None)
        if callable(factory):
            return factory
        cls = getattr(namespace, f"Tracker{method}", None)
        create = getattr(cls, "create", None)
        if callable(create):
            return create
    return None


def available_methods() -> list[str]:
    """Subset of TRACKING_METHODS whose constructor exists in this OpenCV build."""
    return [m for m in TRACKING_METHODS if _constructor(m) is not None]


def create_tracker(method: str):
    """Build a fresh single-object cv2 tracker for the given method name.

    Falls back to the first available method if the requested one isn't built
    into this OpenCV, and raises if no visual tracker is available at all.
    """
    factory = _constructor(method)
    if factory is None:
        for fallback in available_methods():
            factory = _constructor(fallback)
            if factory is not None:
                break
    if factory is None:
        raise RuntimeError(
            "No OpenCV visual tracker is available; install opencv-contrib-python "
            "for CSRT/KCF/MOSSE, or a build that provides TrackerMIL."
        )
    return factory()
