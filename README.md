# FineVideo second index

`run.py` reads a folder of FineVideo Parquet files and writes one JSON line per second. Each line lists which of `person`, `car`, `dog`, `laptop`, and `cell phone` appear, and how many people are visible. YOLO11l runs at 640 with confidence 0.4, class-aware NMS at IoU 0.7, and the Ultralytics letterbox. A rotation tag is applied before inference. A damaged video does not stop the run.

The graded machine is one RTX 4090, 32 threads, and 64 GB of RAM. With only `--input` and `--output`, the script uses `cpu_count - 2` decoder threads and grows the batch until it reaches about 90% of GPU memory. Kaggle's two T4s are only for producing `results.jsonl.gz`. Pass `--workers 2` there so the two processes share the 4 cores.

## Graded install

Ubuntu 22.04, Python 3.11. Internet is available for these commands and is off for the timed run.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libgl1 libglib2.0-0
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prep.py
```

`prep.py` is the preparation command. It downloads `yolo11l.pt` into the working directory. The timed command does not take extra flags:

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

## Kaggle, for the 50 submitted files

The notebook needs a phone-verified account, the GPU T4 x2 accelerator, internet on while downloading, and `HF_TOKEN` set to a Hugging Face read token after you accept the FineVideo terms.

```bash
pip install -r requirements.txt
python prep.py
python kaggle/gate_one_shard.py
```

The gate writes an FP32 reference and an FP16 run for `train-00000-of-01357.parquet`. It exits unless object micro-F1 is at least 0.98, the people count matches exactly on at least 92% of seconds, and it is within 1 on at least 98%. It also checks that a truncated video produces an error line and the process continues. If it prints `GATE FAIL`, set `USE_HALF = False` in `pipeline/runner.py` and produce the 50 files in FP32. Speed does not count until the grader's thresholds pass.

Then turn notebook internet off and rerun that shard with `run.py`. That is the offline rehearsal. After it matches, turn internet back on and process the 50 files:

```bash
python kaggle/run_two.py
```

Each GPU takes one parquet, writes `partials/<name>.jsonl`, and leaves a `.done` marker. A stopped session resumes at the first shard without a marker. Download the `partials` directory and the `timings.json` file. Do not download the videos. Merge on your machine:

```bash
python merge_results.py --partials partials --output results.jsonl.gz
```

Stop the Kaggle session when you step away. An idle notebook still uses the weekly GPU hours. The quota resets Saturday at 05:00 PKT.

## Checks

```bash
python -m unittest discover -s tests -t .
python validate.py reference.jsonl candidate.jsonl --gate
```

## GitHub

This folder is a local git repo. It is not on GitHub yet, because `gh` is not logged in on this machine. After `gh auth login`:

```bash
git add .
git commit -m "Index FineVideo seconds with YOLO11l"
gh repo create alethia-ai --public --source=. --remote=origin --push
```

Do not commit `.env`, the Hugging Face token, `yolo11l.pt`, or Parquet files. `results.jsonl.gz` does get committed once the 50 files have been linted.

## Write-up

`WRITEUP.md` is yours. Write the answers yourself, without an AI tool drafting or editing them. Keep the stage summary that `run.py` prints: it is the timing evidence for questions 1 and 3.
