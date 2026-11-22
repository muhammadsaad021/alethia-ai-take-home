"""Decode every frame from mp4 bytes.

Timestamps are presentation times relative to the first decoded frame.
A rotation tag, either a display matrix or a container rotate tag, is applied
before the frame is returned. Decode goes through PyAV so every frame keeps a
presentation timestamp. nvdec_available() only reports whether an optional
NVIDIA decoder package is installed; a missing package never changes results.
"""

from __future__ import annotations

import io
from typing import Iterator

import cv2
import numpy as np

from pipeline.letterbox import apply_letterbox, build_ultralytics_letterbox


def nvdec_available() -> bool:
    try:
        import PyNvVideoCodec  # noqa: F401
    except Exception:
        return False
    return True


def rotation_clockwise(frame, stream) -> int:
    """Clockwise degrees in {0, 90, 180, 270} needed to display the frame upright."""
    reported = getattr(frame, "rotation", None)
    if isinstance(reported, (int, float)) and int(reported) % 360 != 0:
        # PyAV reports the display-matrix angle counter-clockwise.
        return (-int(reported)) % 360
    metadata = getattr(stream, "metadata", None) or {}
    raw = metadata.get("rotate")
    if raw is None:
        return 0
    try:
        return int(round(float(raw))) % 360
    except (TypeError, ValueError):
        return 0


def snap_right_angle(degrees: int) -> int:
    return int(round((degrees % 360) / 90.0) * 90) % 360


def apply_clockwise(image: np.ndarray, degrees: int) -> np.ndarray:
    degrees = snap_right_angle(degrees)
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _frame_time(frame, stream, index: int) -> float:
    if getattr(frame, "time", None) is not None:
        return float(frame.time)
    time_base = getattr(stream, "time_base", None)
    if getattr(frame, "pts", None) is not None and time_base is not None:
        return float(frame.pts * time_base)
    rate = getattr(stream, "average_rate", None) or getattr(stream, "guessed_rate", None)
    if rate:
        return index / float(rate)
    return index / 30.0


def iter_raw_frames(mp4: bytes) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (seconds_from_first_frame, upright BGR image) for every frame."""
    import av

    if not mp4:
        raise ValueError("empty video")
    container = av.open(io.BytesIO(mp4), mode="r")
    try:
        if not container.streams.video:
            raise ValueError("no video stream")
        stream = container.streams.video[0]
        # One decoder thread per video. The worker pool is what fills the CPU.
        try:
            stream.thread_type = "AUTO"
            stream.thread_count = 1
        except Exception:
            pass
        first_time: float | None = None
        clockwise: int | None = None
        index = 0
        for frame in container.decode(stream):
            when = _frame_time(frame, stream, index)
            if first_time is None:
                first_time = when
                clockwise = rotation_clockwise(frame, stream)
            relative = when - first_time
            image = frame.to_ndarray(format="bgr24")
            if clockwise:
                image = apply_clockwise(image, clockwise)
            yield relative, image
            index += 1
    finally:
        container.close()


def iter_model_frames(mp4: bytes, letterbox=None) -> Iterator[tuple[float, np.ndarray]]:
    """Yield upright frames already letterboxed for YOLO11l."""
    if letterbox is None:
        letterbox = build_ultralytics_letterbox()
    for timestamp, image in iter_raw_frames(mp4):
        yield timestamp, apply_letterbox(image, letterbox)
