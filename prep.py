"""Download yolo11l.pt. Run this once while the machine still has internet."""

from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    from ultralytics import YOLO

    destination = Path.cwd() / "yolo11l.pt"
    model = YOLO(str(destination if destination.is_file() else "yolo11l.pt"))
    source = Path(getattr(model, "ckpt_path", "") or "")
    if source.is_file() and source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    if not destination.is_file():
        found = list(Path.cwd().rglob("yolo11l.pt"))
        if not found:
            raise SystemExit("yolo11l.pt was not downloaded into the working directory")
        if found[0].resolve() != destination.resolve():
            shutil.copyfile(found[0], destination)
    print(f"ready {destination} ({destination.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
