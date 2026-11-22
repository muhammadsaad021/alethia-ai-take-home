"""Merge per-shard JSONL files, gzip them, and lint the result."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from pipeline.aggregate import TARGET_CLASSES

ALLOWED_NAMES = set(TARGET_CLASSES)


def iter_jsonl_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(path for path in source.glob("*.jsonl") if path.is_file())


def lint_records(records: list[dict]) -> list[str]:
    problems: list[str] = []
    seen: set[tuple[str, int]] = set()
    seconds_by_video: dict[str, set[int]] = {}
    for index, row in enumerate(records, start=1):
        video = row.get("video")
        if not isinstance(video, str) or ":" not in video:
            problems.append(f"line {index} has no parquet:row video id")
            continue
        if "error" in row:
            if not isinstance(row["error"], str) or not row["error"].strip():
                problems.append(f"line {index} has an empty error")
            continue
        if "second" not in row or "objects" not in row or "people" not in row:
            problems.append(f"line {index} is missing second, objects, or people")
            continue
        second = row["second"]
        if not isinstance(second, int) or second < 0:
            problems.append(f"line {index} has a bad second")
            continue
        key = (video, second)
        if key in seen:
            problems.append(f"duplicate {video} second {second}")
        seen.add(key)
        seconds_by_video.setdefault(video, set()).add(second)
        names = row["objects"]
        if not isinstance(names, list) or any(name not in ALLOWED_NAMES for name in names):
            problems.append(f"line {index} has objects outside the five classes")
            continue
        canonical = [name for name in TARGET_CLASSES if name in names]
        if names != canonical:
            problems.append(f"line {index} objects are not in canonical order")
        if not isinstance(row["people"], int) or row["people"] < 0:
            problems.append(f"line {index} has a bad people count")
    for video, seconds in seconds_by_video.items():
        if not seconds:
            continue
        missing = [second for second in range(max(seconds) + 1) if second not in seconds]
        if missing:
            preview = ", ".join(str(second) for second in missing[:8])
            problems.append(f"{video} is missing seconds {preview}")
    return problems


def load_records(files: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                text = raw.strip()
                if text:
                    records.append(json.loads(text))
    return records


def write_outputs(records: list[dict], jsonl_path: Path, gzip_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=True) + "\n" for row in records]
    jsonl_path.write_text("".join(lines), encoding="utf-8")
    with gzip.open(gzip_path, "wt", encoding="utf-8") as handle:
        handle.writelines(lines)


def merge(source: Path, gzip_path: Path, jsonl_path: Path | None = None) -> list[str]:
    files = iter_jsonl_files(source)
    if not files:
        raise SystemExit(f"no jsonl files in {source}")
    records = load_records(files)
    problems = lint_records(records)
    destination = jsonl_path or gzip_path.with_suffix("")
    if destination.suffix != ".jsonl":
        destination = Path(str(gzip_path)[:-3] if str(gzip_path).endswith(".gz") else str(gzip_path) + ".jsonl")
    write_outputs(records, destination, gzip_path)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge shard JSONL files into results.jsonl.gz")
    parser.add_argument("--partials", required=True, help="Directory of *.jsonl shard files, or one jsonl file")
    parser.add_argument("--output", default="results.jsonl.gz")
    args = parser.parse_args(argv)
    gzip_path = Path(args.output)
    problems = merge(Path(args.partials), gzip_path)
    print(f"wrote {gzip_path}")
    if problems:
        print(f"LINT FAIL {len(problems)}")
        for problem in problems[:50]:
            print(problem)
        return 1
    print("LINT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
