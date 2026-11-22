import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from merge_results import lint_records, merge  # noqa: E402
from pipeline.shards import pending_shards, shard_filename  # noqa: E402
from validate import passes_gate, report  # noqa: E402


def _line(video, second, objects, people):
    return {"video": video, "second": second, "objects": objects, "people": people}


class ScoreTests(unittest.TestCase):
    def test_perfect_match_passes_the_gate(self):
        rows = [
            _line("train-00000-of-01357.parquet:0", 0, ["person", "car"], 2),
            _line("train-00000-of-01357.parquet:0", 1, [], 0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ("ref.jsonl", "pred.jsonl"):
                (folder / name).write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            result = report(folder / "ref.jsonl", folder / "pred.jsonl")
        self.assertTrue(passes_gate(result))
        self.assertEqual(result["people_exact"], 1.0)
        self.assertAlmostEqual(result["f1"], 1.0)

    def test_missing_object_lowers_f1_and_people(self):
        video = "train-00000-of-01357.parquet:0"
        reference = [_line(video, 0, ["person"], 3), _line(video, 1, ["person"], 3)]
        prediction = [_line(video, 0, [], 0)]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "ref.jsonl").write_text("\n".join(json.dumps(row) for row in reference) + "\n", encoding="utf-8")
            (folder / "pred.jsonl").write_text(json.dumps(prediction[0]) + "\n", encoding="utf-8")
            result = report(folder / "ref.jsonl", folder / "pred.jsonl")
        self.assertFalse(passes_gate(result))
        self.assertLess(result["f1"], 0.98)
        self.assertEqual(result["people_exact"], 0.0)
        self.assertEqual(result["people_within_1"], 0.0)

    def test_lint_rejects_gaps_duplicates_and_bad_names(self):
        video = "train-00003-of-01357.parquet:4"
        problems = lint_records(
            [
                _line(video, 0, ["laptop", "person"], 1),
                _line(video, 0, ["person"], 1),
                _line(video, 2, ["truck"], 0),
            ]
        )
        text = "\n".join(problems)
        self.assertIn("canonical", text)
        self.assertIn("duplicate", text)
        self.assertIn("outside", text)
        self.assertIn("missing", text)

    def test_merge_gzips_and_lints_a_clean_shard(self):
        video = "train-00000-of-01357.parquet:0"
        rows = [_line(video, 0, ["person"], 1), _line(video, 1, [], 0)]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            partials = folder / "partials"
            partials.mkdir()
            (partials / "a.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            gzip_path = folder / "results.jsonl.gz"
            problems = merge(partials, gzip_path)
            self.assertEqual(problems, [])
            with gzip.open(gzip_path, "rt", encoding="utf-8") as handle:
                loaded = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(loaded, rows)

    def test_pending_shards_skip_done_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            partials = Path(tmp)
            first = shard_filename(0)
            (partials / f"{first}.done").write_text("ok\n", encoding="utf-8")
            pending = pending_shards(partials)
        self.assertEqual(len(pending), 49)
        self.assertNotIn(first, pending)
        self.assertEqual(pending[0], shard_filename(1))


if __name__ == "__main__":
    unittest.main()
