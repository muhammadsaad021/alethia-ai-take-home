# Write-up



## 1. Your timings (100 words)

How long did your first working version take on the full 50 files, and how long did your final version take? Which single change saved the most time?

My first working version was FP32 at batch 16. I timed it on one shard only (33 videos, 253,918 frames): 8,933 seconds, about 28.5 frames/sec. I never ran that version on all 50 files, so I can't give a full-50 time for it. My final version uses FP16 with batch 48. It took 4,246 seconds on the same shard (about 60 frames/sec), and the logged 50-file run on two T4s took 31 hours 23 minutes. The biggest saving came from FP16 plus the larger batch it allowed, which cut the shard time by more than half.

## 2. Your pipeline (150 words)

Describe what your script does, step by step, from opening a Parquet file to writing an output line.

The script opens each Parquet file in filename order and reads only the mp4 bytes. The video id is the file name, a colon, and the row number from 0. Several decoder threads decode every frame with PyAV. Timestamps are measured from the first frame, rotation or display-matrix tags are applied, and frames are letterboxed to 640. The main thread runs YOLO11l in batches (FP16 via quantize=16) with confidence 0.4 and class-aware NMS at IoU 0.7. It keeps only person, car, dog, laptop and cell phone. Each frame goes into second floor(timestamp). An object counts as present if it appears in at least half that second's frames, and the people count is the median of per-frame person counts, rounding down at the halfway point. Every second from 0 to the last frame is written, including empty ones. An undecodable video writes an error line. A video that fails midway keeps its seconds and adds one error line.

## 3. Where the time goes (150 words)

Which part of your final version takes the most time? How did you measure this?

GPU inference takes almost all the time. In the FP16 shard run, GPU time was 4,234 of 4,246 seconds. Decoder threads used 411 seconds in total, but they finished well ahead of the GPU, so it waited only 0.7 seconds. I measured this with the stage counters run.py prints at the end of each run (GPU time, GPU wait, decode-thread time), not by guessing. The FP32 run showed the same pattern: 8,919 seconds of GPU time against 1.3 seconds of waiting. So on the T4, decoding is not the bottleneck. Inference is.

## 4. Checking correctness (100 words)

How did you check that your results stayed correct after you made the script faster?

I kept an FP32, every-frame run of one shard as the reference and compared the FP16 output to it with validate.py. Object micro-F1 was 0.9999. People counts matched exactly on 99.81% of seconds and were within 1 on 100% of seconds, with no error videos. I also tested a truncated video: it wrote one error line and the run continued, and the shard finished with zero unexpected errors. Finally, I linted the 50-file results.jsonl.gz for duplicate seconds, gaps and illegal class names. It has 1,650 videos and 461,757 second lines, and the lint found no problems.

## 5. Working with AI (100 words)

Describe one suggestion from an AI tool that was wrong or that you decided not to use. How did you find out it was wrong, or why did you reject it?

When I moved to Kaggle before starting with Modal, an AI tool gave me wrong directions for navigating the environment. It mixed up /kaggle/input and /kaggle/working, treating them as interchangeable. In fact, input is where the dataset is mounted read-only, and working is where outputs are written. Following its steps left me confused about where files were and where results should go. I noticed when the paths didn't behave the way it described. I then searched online to confirm how Kaggle's folders work and prompted the AI again with that correction, which gave me directions that worked.

## 6. Next steps (100 words)

If you had one more week, what would you change, and why?

I'd measure the same stage summary on the 4090. The batch size and decoder thread count there differ from my T4 setup, so my T4 conclusion (GPU-bound, decode never waiting) might not hold. If the GPU still dominates, I'd try a faster inference kernel. Any such change would be checked the same way as the FP16 gate: compare against the FP32 reference with validate.py and require object F1 and people-count agreement to stay at the current level. That way I'd only add speed I can show is correct.



## Reference
All data used here are in timings(2).json, gate one shard result.txt and results.jsonl.gz
