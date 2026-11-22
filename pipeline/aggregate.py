"""Second bucketing, object presence, and people median.

The rules match the take-home brief:

- Second S holds frames with timestamp in [S, S+1), measured from the first
  decoded frame.
- An object is present when it appears in at least half of that second's frames.
- The people count is the median of the per-frame person counts. An even count
  uses integer division, which rounds the halfway case down.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

TARGET_CLASSES: tuple[str, ...] = ("person", "car", "dog", "laptop", "cell phone")
CLASS_IDS: dict[str, int] = {
    "person": 0,
    "car": 2,
    "dog": 16,
    "laptop": 63,
    "cell phone": 67,
}
CLASS_BY_ID: dict[int, str] = {index: name for name, index in CLASS_IDS.items()}
CLASS_ID_LIST: list[int] = [CLASS_IDS[name] for name in TARGET_CLASSES]


def second_index(timestamp: float) -> int:
    """Floor a timestamp that has already been zeroed on the first frame."""
    if timestamp < 0:
        timestamp = 0.0
    return int(timestamp)


def median_half_down(values: list[int]) -> int:
    """Median. Halfway between two integers rounds down."""
    if not values:
        return 0
    ordered = sorted(values)
    count = len(ordered)
    mid = count // 2
    if count % 2 == 1:
        return int(ordered[mid])
    return int(ordered[mid - 1] + ordered[mid]) // 2


def objects_present(hits: dict[str, int], frames: int) -> list[str]:
    """Classes seen in at least half of the frames, in canonical order."""
    if frames <= 0:
        return []
    return [name for name in TARGET_CLASSES if hits.get(name, 0) * 2 >= frames]


@dataclass
class SecondBucket:
    frames: int = 0
    hits: dict[str, int] = field(default_factory=lambda: {name: 0 for name in TARGET_CLASSES})
    people: list[int] = field(default_factory=list)

    def add(self, names: set[str], person_count: int) -> None:
        self.frames += 1
        self.people.append(int(person_count))
        for name in names:
            if name in self.hits:
                self.hits[name] += 1


def success_record(video: str, second: int, names: list[str], people: int) -> dict:
    return {
        "video": video,
        "second": int(second),
        "objects": list(names),
        "people": int(people),
    }


def error_record(video: str, message: str) -> dict:
    text = " ".join((message or "could not decode video").split())
    return {"video": video, "error": text[:180] or "could not decode video"}


def dump_record(record: dict) -> str:
    return json.dumps(record, ensure_ascii=True)


def lines_for_video(
    video: str,
    buckets: dict[int, SecondBucket],
    error: str | None,
) -> list[str]:
    """Build every output line for one video.

    A video with no decoded frames produces only an error line. A video that
    fails after some frames keeps the seconds already counted, including empty
    seconds inside that range, and then one error line.
    """
    lines: list[str] = []
    populated = [second for second, bucket in buckets.items() if bucket.frames > 0]
    if populated:
        last = max(populated)
        for second in range(last + 1):
            bucket = buckets.get(second)
            if bucket is None or bucket.frames == 0:
                record = success_record(video, second, [], 0)
            else:
                record = success_record(
                    video,
                    second,
                    objects_present(bucket.hits, bucket.frames),
                    median_half_down(bucket.people),
                )
            lines.append(dump_record(record))
    if error:
        lines.append(dump_record(error_record(video, error)))
    elif not populated:
        lines.append(dump_record(error_record(video, "could not decode video")))
    return lines


def short_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    text = " ".join(text.split())
    return text[:180] or "could not decode video"
