"""Read FineVideo Parquet files in row order."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq


def list_parquet_files(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix != ".parquet":
            raise ValueError(f"{source} is not a .parquet file")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"input not found: {source}")
    files = sorted(path for path in source.iterdir() if path.suffix == ".parquet" and path.is_file())
    if not files:
        raise FileNotFoundError(f"no .parquet files in {source}")
    return files


def iter_videos(path: Path):
    """Yield (video_id, mp4_bytes_or_None) with the row number starting at 0."""
    with path.open("rb") as handle:
        if handle.read(4) != b"PAR1":
            raise ValueError("not a parquet file")
    name = path.name
    table = pq.read_table(path, columns=["mp4"])
    column = table.column("mp4")
    for index in range(len(column)):
        yield f"{name}:{index}", column[index].as_py()
