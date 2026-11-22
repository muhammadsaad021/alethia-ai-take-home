# Hardware

The 50-file results are produced on a Kaggle notebook, not on the machine that grades the script. These figures were read in that session on Thursday 19 November 2026.

- GPUs: two NVIDIA Tesla T4s. `nvidia-smi` reports 15360 MiB on each card. The accelerator is Kaggle's "GPU T4 x2". One quota hour covers both cards.
- Driver: NVIDIA 580.178.04. The driver reports CUDA 13.0.
- CPU: 4 logical processors (`nproc` printed 4).
- RAM: 31 Gi total, about 29 Gi available (`free -h`). No swap.
- Disk: the notebook's writable files are on `/kaggle/working`. Kaggle keeps about 20 GB of that directory. Each shard is downloaded, indexed, and deleted, because the 50 Parquet files are about 25 GB together.
- OS: `Linux e0a8e4c599f6 6.18.55+ #1 SMP Thu Oct 22 16:37:21 UTC 2026 x86_64 GNU/Linux`.
- Script settings on that machine: `--workers 2` on each T4, FP16, batch chosen from the memory on the card.

`run.py` does not hard-code those limits. On the grading VM it reads the CPU count and the GPU memory, so the same file uses about 30 decoder threads and a larger batch on the RTX 4090 (24 GB, 32 threads, 64 GB RAM).

## Duration

Not measured yet. `kaggle/run_two.py` appends each pair of shards to `/kaggle/working/finevideo_run/timings.json`. After the 50 files finish, add the start-to-finish wall time here, including every resume. Do not invent the number from a single shard.
