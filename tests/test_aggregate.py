import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.aggregate import (  # noqa: E402
    SecondBucket,
    lines_for_video,
    median_half_down,
    objects_present,
    second_index,
)
from pipeline.decode import apply_clockwise  # noqa: E402
from pipeline.letterbox import letterbox_bgr  # noqa: E402


class AggregateTests(unittest.TestCase):
    def test_second_index_floors_and_clamps(self):
        self.assertEqual(second_index(0.0), 0)
        self.assertEqual(second_index(0.999), 0)
        self.assertEqual(second_index(1.0), 1)
        self.assertEqual(second_index(12.2), 12)
        self.assertEqual(second_index(-0.01), 0)

    def test_presence_is_at_least_half(self):
        hits = {"person": 1, "car": 0, "dog": 0, "laptop": 1, "cell phone": 0}
        self.assertEqual(objects_present(hits, 1), ["person", "laptop"])
        self.assertEqual(objects_present({"person": 1, "car": 0, "dog": 0, "laptop": 0, "cell phone": 0}, 2), ["person"])
        self.assertEqual(objects_present({"person": 1, "car": 0, "dog": 0, "laptop": 0, "cell phone": 0}, 3), [])
        self.assertEqual(objects_present(hits, 0), [])

    def test_median_rounds_halfway_down(self):
        self.assertEqual(median_half_down([]), 0)
        self.assertEqual(median_half_down([4]), 4)
        self.assertEqual(median_half_down([1, 2, 9]), 2)
        self.assertEqual(median_half_down([1, 2, 3, 4]), 2)
        self.assertEqual(median_half_down([0, 5]), 2)
        self.assertEqual(median_half_down([2, 2, 2, 4]), 2)

    def test_object_order_and_empty_second(self):
        buckets = {
            0: SecondBucket(frames=2, hits={"person": 2, "car": 0, "dog": 1, "laptop": 2, "cell phone": 0}, people=[1, 3]),
            2: SecondBucket(frames=1, hits={"person": 0, "car": 0, "dog": 0, "laptop": 0, "cell phone": 1}, people=[0]),
        }
        lines = [json.loads(line) for line in lines_for_video("train-00003-of-01357.parquet:4", buckets, None)]
        self.assertEqual([row["second"] for row in lines], [0, 1, 2])
        self.assertEqual(lines[0]["objects"], ["person", "dog", "laptop"])
        self.assertEqual(lines[0]["people"], 2)
        self.assertEqual(lines[1]["objects"], [])
        self.assertEqual(lines[1]["people"], 0)
        self.assertEqual(lines[2]["objects"], ["cell phone"])
        self.assertEqual(lines[2]["video"], "train-00003-of-01357.parquet:4")

    def test_error_only_and_partial_error(self):
        only = [json.loads(line) for line in lines_for_video("train-00000-of-01357.parquet:1", {}, "could not decode video")]
        self.assertEqual(only, [{"video": "train-00000-of-01357.parquet:1", "error": "could not decode video"}])
        bucket = SecondBucket(frames=1, hits={"person": 1, "car": 0, "dog": 0, "laptop": 0, "cell phone": 0}, people=[2])
        both = [json.loads(line) for line in lines_for_video("clip.parquet:0", {0: bucket}, "decoder stopped")]
        self.assertEqual(both[0]["people"], 2)
        self.assertIn("error", both[1])
        self.assertNotIn("second", both[1])

    def test_letterbox_is_640_and_padded(self):
        image = np.full((32, 16, 3), 10, dtype=np.uint8)
        boxed = letterbox_bgr(image)
        self.assertEqual(boxed.shape, (640, 640, 3))
        self.assertEqual(tuple(int(v) for v in boxed[0, 0]), (114, 114, 114))
        self.assertEqual(tuple(int(v) for v in boxed[320, 320]), (10, 10, 10))

    def test_clockwise_rotation_moves_the_marked_pixel(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[0, 1] = (9, 8, 7)
        turned = apply_clockwise(image, 90)
        self.assertTrue(np.array_equal(turned[1, 1], np.array([9, 8, 7], dtype=np.uint8)))
        self.assertTrue(np.array_equal(apply_clockwise(image, 0), image))


if __name__ == "__main__":
    unittest.main()
