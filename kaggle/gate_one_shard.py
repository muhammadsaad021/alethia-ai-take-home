"""Prove one shard on a Kaggle T4 before spending the weekly quota.

This downloads train-00000, writes an FP32 reference and an FP16 run, and
refuses to continue unless validate.py clears the internal margin
(micro-F1 >= 0.98, people exact >= 0.92, within 1 >= 0.98). It then runs a
truncated video and checks that the process stays alive and writes an error line.

After this passes, turn the notebook internet off and run, from the repo root:

    python run.py --input <shard dir> --output offline.jsonl --workers 2

That is the offline rehearsal of the graded command. prep.py must already have
left yolo11l.pt in the repo root.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kaggle.run_two import download_shard, token, work_root  # noqa: E402
from pipeline.shards import shard_filename  # noqa: E402


def run_command(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    if token() is None:
        raise SystemExit("Set HF_TOKEN before the one-shard gate.")
    weights = ROOT / "yolo11l.pt"
    if not weights.is_file():
        raise SystemExit("Missing yolo11l.pt. Run python prep.py first.")
    root = work_root() / "gate"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    name = shard_filename(0)
    downloaded = download_shard(name, root / "cache")
    shard_dir = root / "shard"
    shard_dir.mkdir()
    shard_path = shard_dir / name
    shutil.move(str(downloaded), shard_path)

    reference = root / "reference.jsonl"
    fast = root / "fast.jsonl"
    common = [
        sys.executable,
        str(ROOT / "run.py"),
        "--input",
        str(shard_dir),
        "--weights",
        str(weights),
        "--workers",
        "2",
    ]
    run_command(common + ["--fp32", "--output", str(reference)])
    run_command(common + ["--output", str(fast)])
    run_command([sys.executable, str(ROOT / "validate.py"), str(reference), str(fast), "--gate"])

    damaged_dir = root / "damaged"
    damaged_dir.mkdir()
    table = pa.table({"mp4": pa.array([b"\x00\x00not-a-video"], type=pa.large_binary())})
    damaged_name = "train-00999-of-01357.parquet"
    pq.write_table(table, damaged_dir / damaged_name)
    damaged_out = root / "damaged.jsonl"
    run_command(
        [
            sys.executable,
            str(ROOT / "run.py"),
            "--input",
            str(damaged_dir),
            "--output",
            str(damaged_out),
            "--weights",
            str(weights),
            "--workers",
            "1",
        ]
    )
    rows = [json.loads(line) for line in damaged_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not any(row.get("video") == f"{damaged_name}:0" and "error" in row for row in rows):
        raise SystemExit(f"damaged video did not produce an error line: {rows}")
    print("DAMAGED PASS")
    print("Gate passed. Turn notebook internet OFF and rerun one shard with run.py before the full 50.")


if __name__ == "__main__":
    main()
