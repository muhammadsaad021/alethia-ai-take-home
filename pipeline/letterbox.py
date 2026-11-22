"""Letterbox a BGR frame to 640x640 the way Ultralytics does.

auto is off, so every frame becomes exactly 640x640. Scale-up is allowed,
padding is centered, and the pad color is 114. A later Ultralytics letterbox
of an already square 640 image is a no-op, so box coordinates are irrelevant:
only class presence and the person count are read back.
"""

from __future__ import annotations

import cv2
import numpy as np

IMGSZ = 640
PAD_COLOR = (114, 114, 114)


def letterbox_bgr(image: np.ndarray, new_shape: tuple[int, int] = (IMGSZ, IMGSZ)) -> np.ndarray:
    shape = image.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    resized = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    pad_w = (new_shape[1] - resized[0]) / 2
    pad_h = (new_shape[0] - resized[1]) / 2
    if (shape[1], shape[0]) != resized:
        image = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)
    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))
    return cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=PAD_COLOR,
    )


def build_ultralytics_letterbox():
    """Return Ultralytics' own LetterBox when the package is installed."""
    try:
        from ultralytics.data.augment import LetterBox
    except Exception:
        return None
    attempts = (
        {"new_shape": (IMGSZ, IMGSZ), "auto": False, "scaleup": True, "center": True, "stride": 32},
        {"new_shape": (IMGSZ, IMGSZ), "auto": False, "scaleup": True, "stride": 32},
    )
    for kwargs in attempts:
        try:
            return LetterBox(**kwargs)
        except TypeError:
            continue
    try:
        return LetterBox((IMGSZ, IMGSZ), False, False, True, 32)
    except TypeError:
        return None


def apply_letterbox(image: np.ndarray, box=None) -> np.ndarray:
    if box is not None:
        try:
            out = box(image=image)
        except TypeError:
            out = box(image)
        if isinstance(out, dict):
            out = out.get("img", image)
        if isinstance(out, np.ndarray):
            return out
    return letterbox_bgr(image)
