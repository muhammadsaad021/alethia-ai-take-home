"""Index every second of every FineVideo parquet in a folder.

Graded command, with no extra flags:

    python run.py --input <folder of .parquet files> --output results.jsonl

The process sizes itself from the machine it finds. On the grading VM that is
about 30 decoder threads and a batch that fills the 24 GB RTX 4090. On a
Kaggle T4, pass --workers 2 so two processes can share the 4 cores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.runner import execute


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index FineVideo seconds with YOLO11l")
    parser.add_argument("--input", required=True, help="Folder of .parquet files, or one .parquet file")
    parser.add_argument("--output", required=True, help="JSON Lines output path")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Decoder threads. Default is cpu_count - 2.",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Full-precision inference, for checking the FP16 path against a reference.",
    )
    parser.add_argument("--weights", default="yolo11l.pt", help="Local yolo11l.pt path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    execute(
        source=Path(args.input),
        output_path=Path(args.output),
        weights=Path(args.weights),
        workers=args.workers,
        fp32=args.fp32,
    )


if __name__ == "__main__":
    main()
