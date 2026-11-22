"""Compare two JSONL files the way the grader scores a run.

Object score is micro-F1 over the five classes on every (video, second).
People score is the fraction of seconds that match exactly, and the fraction
within 1. Missing seconds count as an empty prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.aggregate import TARGET_CLASSES

GATE_F1 = 0.98
GATE_EXACT = 0.92
GATE_WITHIN = 0.98


def load_jsonl(path: Path) -> tuple[dict[tuple[str, int], dict], dict[str, str]]:
    seconds: dict[tuple[str, int], dict] = {}
    errors: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            row = json.loads(text)
            video = row["video"]
            if "error" in row:
                errors[video] = str(row["error"])
                continue
            seconds[(video, int(row["second"]))] = row
    return seconds, errors


def _sets(table: dict[tuple[str, int], dict], key: tuple[str, int]) -> set[str]:
    row = table.get(key)
    if row is None:
        return set()
    return set(row.get("objects") or [])


def micro_f1(reference: dict, prediction: dict) -> dict[str, float]:
    totals = {name: {"tp": 0, "fp": 0, "fn": 0} for name in TARGET_CLASSES}
    keys = set(reference) | set(prediction)
    for key in keys:
        truth = _sets(reference, key)
        guess = _sets(prediction, key)
        for name in TARGET_CLASSES:
            cell = totals[name]
            if name in truth and name in guess:
                cell["tp"] += 1
            elif name in guess:
                cell["fp"] += 1
            elif name in truth:
                cell["fn"] += 1
    tp = sum(cell["tp"] for cell in totals.values())
    fp = sum(cell["fp"] for cell in totals.values())
    fn = sum(cell["fn"] for cell in totals.values())
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 1.0
    per_class = {}
    for name, cell in totals.items():
        p = cell["tp"] / (cell["tp"] + cell["fp"]) if cell["tp"] + cell["fp"] else 1.0
        r = cell["tp"] / (cell["tp"] + cell["fn"]) if cell["tp"] + cell["fn"] else 1.0
        per_class[name] = 2 * p * r / (p + r) if p + r else 1.0
    return {"f1": score, "precision": precision, "recall": recall, "per_class": per_class}


def people_rates(reference: dict, prediction: dict) -> tuple[float, float, int]:
    keys = set(reference) | set(prediction)
    if not keys:
        return 1.0, 1.0, 0
    exact = 0
    within = 0
    for key in keys:
        truth = int(reference[key]["people"]) if key in reference else 0
        guess = int(prediction[key]["people"]) if key in prediction else 0
        if truth == guess:
            exact += 1
        if abs(truth - guess) <= 1:
            within += 1
    count = len(keys)
    return exact / count, within / count, count


def report(reference_path: Path, prediction_path: Path) -> dict:
    reference, reference_errors = load_jsonl(reference_path)
    prediction, prediction_errors = load_jsonl(prediction_path)
    scored_ref = {key: row for key, row in reference.items() if key[0] not in reference_errors}
    scored_pred = {key: row for key, row in prediction.items() if key[0] not in reference_errors}
    scores = micro_f1(scored_ref, scored_pred)
    exact, within, count = people_rates(scored_ref, scored_pred)
    return {
        "seconds": count,
        "f1": scores["f1"],
        "precision": scores["precision"],
        "recall": scores["recall"],
        "per_class": scores["per_class"],
        "people_exact": exact,
        "people_within_1": within,
        "reference_errors": len(reference_errors),
        "prediction_errors": len(prediction_errors),
    }


def passes_gate(result: dict, f1: float = GATE_F1, exact: float = GATE_EXACT, within: float = GATE_WITHIN) -> bool:
    return result["f1"] >= f1 and result["people_exact"] >= exact and result["people_within_1"] >= within


def print_report(result: dict) -> None:
    print(f"seconds {result['seconds']}")
    print(f"object_micro_f1 {result['f1']:.4f}")
    print(f"object_precision {result['precision']:.4f}")
    print(f"object_recall {result['recall']:.4f}")
    for name, score in result["per_class"].items():
        print(f"f1_{name.replace(' ', '_')} {score:.4f}")
    print(f"people_exact {result['people_exact']:.4f}")
    print(f"people_within_1 {result['people_within_1']:.4f}")
    print(f"reference_error_videos {result['reference_errors']}")
    print(f"prediction_error_videos {result['prediction_errors']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare a candidate JSONL with a reference JSONL")
    parser.add_argument("reference")
    parser.add_argument("prediction")
    parser.add_argument("--gate", action="store_true", help="Exit 1 unless the internal margin above the grader is met")
    args = parser.parse_args(argv)
    result = report(Path(args.reference), Path(args.prediction))
    print_report(result)
    if args.gate and not passes_gate(result):
        print(
            f"GATE FAIL need f1>={GATE_F1} people_exact>={GATE_EXACT} people_within_1>={GATE_WITHIN}",
            file=sys.stderr,
        )
        return 1
    if args.gate:
        print("GATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
