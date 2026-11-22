"""Run the 50 FineVideo shards on Kaggle's two T4 GPUs.

Each process calls the same run.py the grader runs. A shard is skipped only
when its .done marker exists, so a killed session resumes at the next file.
Download one parquet per GPU, delete it after the JSONL is flushed, and stop
the notebook when you step away so the weekly quota is not burned idle.

    python kaggle/run_two.py

Requires HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) and yolo11l.pt from prep.py.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.shards import done_marker, pending_shards, remote_filename, REPO_ID  # noqa: E402


def work_root() -> Path:
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working / "finevideo_run"
    return ROOT / "kaggle_work"


def token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def download_shard(name: str, cache: Path) -> Path:
    from huggingface_hub import hf_hub_download

    cache.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(
        repo_id=REPO_ID,
        filename=remote_filename(name),
        repo_type="dataset",
        local_dir=cache,
        token=token(),
    )
    return Path(local)


def launch(name: str, gpu: str, root: Path, workers: int) -> subprocess.Popen:
    shard_dir = root / "inbox" / f"gpu{gpu}"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True)
    source = download_shard(name, root / "cache" / name)
    destination = shard_dir / name
    shutil.move(str(source), destination)
    partials = root / "partials"
    partials.mkdir(parents=True, exist_ok=True)
    output = partials / f"{name}.jsonl"
    if output.exists():
        output.unlink()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    command = [
        sys.executable,
        str(ROOT / "run.py"),
        "--input",
        str(shard_dir),
        "--output",
        str(output),
        "--workers",
        str(workers),
        "--weights",
        str(ROOT / "yolo11l.pt"),
    ]
    return subprocess.Popen(command, cwd=ROOT, env=env)


def record(root: Path, row: dict) -> None:
    path = root / "timings.json"
    rows = []
    if path.is_file():
        rows = json.loads(path.read_text(encoding="utf-8"))
    rows.append(row)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def run_batch(names: list[str], root: Path, workers: int) -> None:
    started = time.perf_counter()
    processes = []
    try:
        for gpu, name in enumerate(names):
            processes.append((str(gpu), name, launch(name, str(gpu), root, workers)))
    except Exception:
        for _, _, process in processes:
            process.kill()
        raise
    failed = False
    for gpu, name, process in processes:
        code = process.wait()
        partial = root / "partials" / f"{name}.jsonl"
        inbox = root / "inbox" / f"gpu{gpu}"
        if code == 0 and partial.is_file() and partial.stat().st_size > 0:
            done_marker(root / "partials", name).write_text("ok\n", encoding="utf-8")
        else:
            failed = True
            print(f"FAILED {name} exit {code}", file=sys.stderr)
        if inbox.exists():
            shutil.rmtree(inbox, ignore_errors=True)
        cache = root / "cache" / name
        if cache.exists():
            shutil.rmtree(cache, ignore_errors=True)
    record(
        root,
        {
            "shards": names,
            "seconds": round(time.perf_counter() - started, 3),
            "ok": not failed,
        },
    )
    if failed:
        raise SystemExit(1)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Process FineVideo shards on two Kaggle T4s")
    parser.add_argument("--workers", type=int, default=2, help="Decoder threads per GPU")
    parser.add_argument("--gpus", type=int, default=2, help="How many shards to run at once")
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many shards. 0 means all remaining.")
    args = parser.parse_args()
    if token() is None:
        raise SystemExit("Set HF_TOKEN before downloading FineVideo.")
    root = work_root()
    root.mkdir(parents=True, exist_ok=True)
    remaining = pending_shards(root / "partials")
    if args.limit > 0:
        remaining = remaining[: args.limit]
    print(f"pending {len(remaining)} -> {root}")
    while remaining:
        batch = remaining[: max(1, args.gpus)]
        remaining = remaining[max(1, args.gpus) :]
        print("starting", " ".join(batch))
        run_batch(batch, root, args.workers)
    print("all requested shards have .done markers")


if __name__ == "__main__":
    main()
