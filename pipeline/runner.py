"""Overlap decode and letterbox on the CPU with batched GPU inference.

Worker count defaults to cpu_count - 2, so a 32-thread grader uses about 30
decoder threads and one thread owns the GPU. The frame queue is bounded, so a
fast GPU applies backpressure instead of storing a whole parquet of frames.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.aggregate import SecondBucket, lines_for_video, second_index, short_error
from pipeline.decode import iter_model_frames, nvdec_available
from pipeline.infer import (
    choose_batch_size,
    cuda_device,
    device_label,
    load_model,
    predict_batch,
    resolve_weights,
    summarize_result,
)
from pipeline.letterbox import build_ultralytics_letterbox
from pipeline.parquet_io import iter_videos, list_parquet_files


def default_workers() -> int:
    cores = os.cpu_count() or 1
    return max(1, cores - 2)


# The graded command uses FP16. Set this false if the Kaggle gate prints GATE FAIL.
USE_HALF = True


@dataclass
class VideoState:
    video_id: str
    buckets: dict[int, SecondBucket] = field(default_factory=dict)
    error: str | None = None
    expected: int | None = None
    inferred: int = 0
    written: bool = False

    def add_frame(self, timestamp: float, names: set[str], people: int) -> None:
        second = second_index(timestamp)
        bucket = self.buckets.get(second)
        if bucket is None:
            bucket = SecondBucket()
            self.buckets[second] = bucket
        bucket.add(names, people)


@dataclass
class _Frame:
    video_id: str
    timestamp: float
    image: object


@dataclass
class _End:
    video_id: str
    expected: int
    error: str | None


@dataclass
class RunStats:
    videos: int = 0
    errors: int = 0
    frames: int = 0
    seconds: int = 0
    decode_s: float = 0.0
    queue_wait_s: float = 0.0
    gpu_s: float = 0.0
    wall_s: float = 0.0
    workers: int = 0
    batch_size: int = 0
    half: bool = False
    device: str = ""
    nvdec: bool = False

    def lines(self) -> list[str]:
        infer_rate = self.frames / self.gpu_s if self.gpu_s > 0 else 0.0
        wall_rate = self.frames / self.wall_s if self.wall_s > 0 else 0.0
        return [
            f"videos {self.videos}",
            f"errors {self.errors}",
            f"frames {self.frames}",
            f"seconds {self.seconds}",
            f"batch {self.batch_size}",
            f"workers {self.workers}",
            f"device {self.device}",
            f"half {self.half}",
            f"nvdec_importable {self.nvdec}",
            f"decode_thread_s {self.decode_s:.2f}",
            f"gpu_wait_s {self.queue_wait_s:.2f}",
            f"gpu_s {self.gpu_s:.2f}",
            f"infer_frames_per_s {infer_rate:.1f}",
            f"wall_frames_per_s {wall_rate:.1f}",
        ]


def _write_video(handle, state: VideoState, stats: RunStats) -> None:
    if state.written:
        return
    produced = lines_for_video(state.video_id, state.buckets, state.error)
    for line in produced:
        handle.write(line + "\n")
    handle.flush()
    state.written = True
    stats.videos += 1
    if state.error or not any(bucket.frames for bucket in state.buckets.values()):
        stats.errors += 1
    stats.seconds += sum(1 for line in produced if '"second"' in line)
    print(f"video {stats.videos} {state.video_id}", flush=True)


def _maybe_finish(handle, state: VideoState, stats: RunStats) -> None:
    if state.written or state.expected is None:
        return
    if state.inferred >= state.expected:
        _write_video(handle, state, stats)


def _decode_worker(job_q: queue.Queue, frame_q: queue.Queue, stats: RunStats, lock: threading.Lock) -> None:
    letterbox = build_ultralytics_letterbox()
    while True:
        job = job_q.get()
        if job is None:
            return
        video_id, blob = job
        sent = 0
        error = None
        compute = 0.0
        started = time.perf_counter()
        try:
            if not blob:
                raise ValueError("empty video")
            for timestamp, image in iter_model_frames(blob, letterbox):
                compute += time.perf_counter() - started
                frame_q.put(_Frame(video_id, timestamp, image))
                sent += 1
                started = time.perf_counter()
        except Exception as exc:
            compute += time.perf_counter() - started
            error = short_error(exc)
        else:
            compute += time.perf_counter() - started
        with lock:
            stats.decode_s += compute
        frame_q.put(_End(video_id, sent, error))


def _infer_frames(model, batch: list[_Frame], states: dict[str, VideoState], handle, stats: RunStats, half: bool, device) -> None:
    if not batch:
        return
    started = time.perf_counter()
    try:
        results = predict_batch(model, [item.image for item in batch], half=half, device=device)
    except Exception:
        # A bad frame should not drop the rest of the batch.
        results = []
        for item in batch:
            try:
                results.extend(predict_batch(model, [item.image], half=half, device=device))
            except Exception as exc:
                results.append(exc)
    stats.gpu_s += time.perf_counter() - started
    for item, result in zip(batch, results):
        state = states[item.video_id]
        if isinstance(result, Exception):
            state.error = state.error or short_error(result)
        else:
            names, people = summarize_result(result)
            state.add_frame(item.timestamp, names, people)
            stats.frames += 1
        state.inferred += 1
        _maybe_finish(handle, state, stats)


def run_rows(
    rows: list[tuple[str, bytes | None]],
    model,
    output,
    workers: int,
    half: bool,
    batch_size: int,
    device,
    stats: RunStats | None = None,
) -> RunStats:
    """Index one sequence of (video_id, mp4_bytes) rows into an open text file."""
    stats = stats or RunStats()
    stats.workers = workers
    stats.batch_size = batch_size
    stats.half = bool(half and device != "cpu")
    stats.device = device_label(device) if stats.device == "" else stats.device
    if not rows:
        return stats

    frame_q: queue.Queue = queue.Queue(maxsize=max(64, workers * 8))
    job_q: queue.Queue = queue.Queue()
    states = {video_id: VideoState(video_id) for video_id, _ in rows}
    for row in rows:
        job_q.put(row)
    for _ in range(workers):
        job_q.put(None)

    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=_decode_worker,
            args=(job_q, frame_q, stats, lock),
            name=f"decode-{index}",
            daemon=True,
        )
        for index in range(workers)
    ]
    for thread in threads:
        thread.start()

    batch: list[_Frame] = []
    ends = 0
    try:
        while ends < len(rows):
            waited = time.perf_counter()
            item = frame_q.get()
            stats.queue_wait_s += time.perf_counter() - waited
            if isinstance(item, _Frame):
                batch.append(item)
                if len(batch) >= batch_size:
                    _infer_frames(model, batch, states, output, stats, stats.half, device)
                    batch = []
                continue
            ends += 1
            state = states[item.video_id]
            state.expected = item.expected
            if item.error:
                state.error = item.error
            _maybe_finish(output, state, stats)
        if batch:
            _infer_frames(model, batch, states, output, stats, stats.half, device)
            batch = []
        for state in states.values():
            if state.written:
                continue
            if state.expected is None or state.inferred < state.expected:
                state.error = state.error or "inference ended early"
                state.expected = state.inferred
            _maybe_finish(output, state, stats)
    finally:
        for thread in threads:
            thread.join(timeout=60)
    return stats


def run_parquet_files(
    files: list[Path],
    output_path: Path,
    model,
    workers: int,
    half: bool,
    batch_size: int,
    device,
) -> RunStats:
    stats = RunStats(
        workers=workers,
        batch_size=batch_size,
        half=bool(half and device != "cpu"),
        device=device_label(device),
        nvdec=nvdec_available(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", buffering=1) as handle:
        for path in files:
            try:
                rows = list(iter_videos(path))
            except Exception as exc:
                from pipeline.aggregate import dump_record, error_record

                handle.write(dump_record(error_record(f"{path.name}:0", short_error(exc))) + "\n")
                handle.flush()
                stats.videos += 1
                stats.errors += 1
                continue
            run_rows(rows, model, handle, workers, half, batch_size, device, stats)
    return stats


def execute(
    source: Path,
    output_path: Path,
    weights: Path,
    workers: int | None = None,
    fp32: bool = False,
) -> RunStats:
    import torch

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    device = cuda_device()
    half = USE_HALF and not fp32
    if device == "cpu":
        half = False
    model = load_model(resolve_weights(weights))
    batch_size = choose_batch_size(model, half=half, device=device)
    worker_count = workers if workers and workers > 0 else default_workers()
    files = list_parquet_files(source)
    started = time.perf_counter()
    stats = run_parquet_files(files, output_path, model, worker_count, half, batch_size, device)
    stats.wall_s = time.perf_counter() - started
    elapsed = stats.wall_s
    print(f"elapsed_s {elapsed:.2f}")
    for line in stats.lines():
        print(line)
    if device != "cpu":
        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"peak_gpu_mib {peak:.0f}")
    return stats
