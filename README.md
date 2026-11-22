# FineVideo second index

`run.py` reads a folder of FineVideo Parquet files and writes one JSON line per second. Each line lists which of `person`, `car`, `dog`, `laptop`, and `cell phone` appear, and how many people are visible.

The model is Ultralytics YOLO11l at 640, confidence 0.4, class-aware NMS at IoU 0.7, with the default letterbox. A rotation tag is applied before inference. A damaged video does not stop the run. `results.jsonl.gz` is the merged output for the first 50 training shards, 1,650 videos and 461,757 seconds.

## Graded install

Ubuntu 22.04, Python 3.11. These commands have internet. The timed command does not.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libgl1 libglib2.0-0
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prep.py
```

`python prep.py` is the preparation command. It downloads `yolo11l.pt` into the working directory. Then:

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

No other flags are required. The script uses `cpu_count - 2` decoder threads and chooses a batch that stays under about 90% of GPU memory. On the grading machine that is one RTX 4090, 32 threads, and 64 GB of RAM.

## How the submitted results were produced

The 50 files were indexed on a Kaggle notebook with two Tesla T4 GPUs. See `HARDWARE.md`. The launcher is `kaggle/run_two.py`. It is not imported by `run.py`.
