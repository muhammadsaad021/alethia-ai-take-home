import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.runner import run_rows  # noqa: E402


class _Boxes:
    def __init__(self, classes):
        self.cls = np.asarray(classes, dtype=float)

    def __len__(self):
        return int(self.cls.shape[0])


class _Result:
    def __init__(self, classes):
        self.boxes = _Boxes(classes)


class _Model:
    def predict(self, images, **kwargs):
        return [_Result([0, 0]) for _ in images]


def _video_bytes():
    import av

    buffer = __import__("io").BytesIO()
    container = av.open(buffer, "w", format="mp4")
    stream = container.add_stream("mpeg4", rate=3)
    stream.width = 64
    stream.height = 48
    stream.pix_fmt = "yuv420p"
    for index in range(6):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[:, :] = (20, 40, 60)
        image[0, 0] = (index, 0, 0)
        frame = av.VideoFrame.from_ndarray(image[:, :, ::-1], format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return buffer.getvalue()


class PipelineTests(unittest.TestCase):
    def test_corrupt_parquet_writes_an_error_and_returns(self):
        from pipeline.runner import run_parquet_files

        with __import__("tempfile").TemporaryDirectory() as tmp:
            folder = Path(tmp)
            broken = folder / "train-00008-of-01357.parquet"
            broken.write_bytes(b"not a parquet file")
            output = folder / "out.jsonl"
            run_parquet_files([broken], output, _Model(), workers=1, half=False, batch_size=1, device="cpu")
            parsed = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(parsed[0]["video"], "train-00008-of-01357.parquet:0")
        self.assertIn("error", parsed[0])

    def test_undecodable_bytes_write_one_error_line(self):
        handle = __import__("io").StringIO()
        run_rows(
            [("train-00009-of-01357.parquet:3", b"this is not a video")],
            _Model(),
            handle,
            workers=1,
            half=False,
            batch_size=2,
            device="cpu",
        )
        parsed = [json.loads(line) for line in handle.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["video"], "train-00009-of-01357.parquet:3")
        self.assertIn("error", parsed[0])
        self.assertNotIn("second", parsed[0])

    def test_bad_bytes_become_an_error_and_the_next_video_still_runs(self):
        try:
            good = _video_bytes()
        except Exception as exc:
            self.skipTest(f"PyAV encoder is not available: {exc}")
        rows = [
            ("train-00000-of-01357.parquet:0", b"this is not a video"),
            ("train-00000-of-01357.parquet:1", good),
        ]
        handle = __import__("io").StringIO()
        run_rows(rows, _Model(), handle, workers=2, half=False, batch_size=4, device="cpu")
        parsed = [json.loads(line) for line in handle.getvalue().splitlines() if line.strip()]
        errors = [row for row in parsed if row["video"].endswith(":0")]
        seconds = [row for row in parsed if row["video"].endswith(":1") and "second" in row]
        self.assertEqual(len(errors), 1)
        self.assertIn("error", errors[0])
        self.assertGreaterEqual(len(seconds), 1)
        self.assertEqual(seconds[0]["objects"], ["person"])
        self.assertEqual(seconds[0]["people"], 2)
        self.assertEqual(seconds[0]["second"], 0)
        self.assertEqual([row["second"] for row in seconds], list(range(len(seconds))))


if __name__ == "__main__":
    unittest.main()
