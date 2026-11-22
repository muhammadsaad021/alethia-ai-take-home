"""YOLO11l batch inference with the Ultralytics defaults the brief requires."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.aggregate import CLASS_BY_ID, CLASS_ID_LIST
from pipeline.letterbox import IMGSZ

CONF = 0.4
IOU = 0.7


def resolve_weights(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    raise SystemExit(
        f"Missing weights file {candidate}. Run `python prep.py` once while online, "
        "then start the timed run from the same directory."
    )


def load_model(weights: Path):
    from ultralytics import YOLO

    return YOLO(str(weights))


def cuda_device():
    import torch

    if torch.cuda.is_available():
        return 0
    return "cpu"


def device_label(device) -> str:
    if device == "cpu":
        return "cpu"
    import torch

    props = torch.cuda.get_device_properties(0)
    memory = getattr(props, "total_memory", 0) / (1024 * 1024)
    return f"{props.name} ({memory:.0f} MiB)"


def summarize_result(result) -> tuple[set[str], int]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return set(), 0
    classes = boxes.cls
    if hasattr(classes, "detach"):
        classes = classes.detach().cpu().numpy()
    else:
        classes = np.asarray(classes)
    names: set[str] = set()
    people = 0
    for value in classes.tolist():
        class_id = int(value)
        name = CLASS_BY_ID.get(class_id)
        if name:
            names.add(name)
        if class_id == 0:
            people += 1
    return names, people


def _call_predict(model, images: list[np.ndarray], **kwargs):
    results = model.predict(images, **kwargs)
    if not isinstance(results, list):
        results = list(results)
    return results


def predict_batch(model, images: list[np.ndarray], half: bool, device):
    """Run one batch. FP16 uses quantize=16; the old half flag warns on every frame."""
    use_half = bool(half and device != "cpu")
    kwargs = {
        "imgsz": IMGSZ,
        "conf": CONF,
        "iou": IOU,
        "classes": CLASS_ID_LIST,
        "verbose": False,
        "agnostic_nms": False,
        "device": device,
        "stream": False,
        "rect": False,
    }
    try:
        return _call_predict(model, images, quantize=16 if use_half else 32, **kwargs)
    except TypeError:
        return _call_predict(model, images, half=use_half, **kwargs)


def choose_batch_size(model, half: bool, device, cap: int = 64) -> int:
    """Largest batch that stays under about 90% of GPU memory.

    The search runs inside the timed command, so it stops at the first size
    that fills the card or at `cap`. CPU runs stay at batch 1.
    """
    if device == "cpu":
        return 1
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    chosen = 1
    print("probing batch size", flush=True)
    for size in (8, 16, 32, 48):
        if size > cap:
            break
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f"probing batch {size}", flush=True)
            predict_batch(model, [dummy] * size, half=half, device=device)
            used = torch.cuda.max_memory_allocated()
            chosen = size
            print(f"batch {size} fits ({used / (1024 * 1024):.0f} MiB)", flush=True)
            if used >= int(total * 0.90):
                break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            print(f"batch {size} out of memory, using {chosen}", flush=True)
            break
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"using batch {chosen}", flush=True)
    return chosen
