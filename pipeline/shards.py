"""The first 50 FineVideo training shards."""

from __future__ import annotations

from pathlib import Path

SHARD_COUNT = 50
REPO_ID = "HuggingFaceFV/finevideo"


def shard_filename(index: int) -> str:
    return f"train-{index:05d}-of-01357.parquet"


def all_shard_filenames() -> list[str]:
    return [shard_filename(index) for index in range(SHARD_COUNT)]


def remote_filename(name: str) -> str:
    return f"data/{name}"


def done_marker(partials: Path, name: str) -> Path:
    return partials / f"{name}.done"


def pending_shards(partials: Path) -> list[str]:
    return [name for name in all_shard_filenames() if not done_marker(partials, name).is_file()]
