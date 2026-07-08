#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from shadow_to_think.data_io import read_jsonl, write_jsonl
from shadow_to_think.verifier import verify_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare local and shadow decoding outputs against gold answers.")
    p.add_argument("--dataset_path", action="append", required=True, help="Gold JSONL. Repeat for shards.")
    p.add_argument("--local_path", action="append", required=True, help="Local decode JSONL. Repeat for shards.")
    p.add_argument("--shadow_path", action="append", required=True, help="Shadow decode JSONL. Repeat for shards.")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def _rows_by_id(paths: Iterable[str]) -> Dict[str, Dict]:
    rows: Dict[str, Dict] = {}
    for path in paths:
        for row in read_jsonl(path):
            row_id = row.get("id")
            if row_id is None:
                raise ValueError(f"Missing id in {path}")
            if row_id in rows:
                raise ValueError(f"Duplicate id {row_id!r}")
            rows[str(row_id)] = row
    return rows


def _outcome(local_correct: bool, shadow_correct: bool) -> str:
    if local_correct and shadow_correct:
        return "kept_correct"
    if not local_correct and shadow_correct:
        return "fixed"
    if local_correct and not shadow_correct:
        return "broke"
    return "missed_wrong"


def main() -> None:
    args = parse_args()
    if not (len(args.dataset_path) == len(args.local_path) == len(args.shadow_path)):
        raise ValueError("--dataset_path, --local_path, and --shadow_path must have the same count")

    gold_rows = _rows_by_id(args.dataset_path)
    local_rows = _rows_by_id(args.local_path)
    shadow_rows = _rows_by_id(args.shadow_path)
    missing = (set(gold_rows) - set(local_rows)) | (set(gold_rows) - set(shadow_rows))
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise ValueError(f"Missing decode rows for {len(missing)} ids, e.g. {sample}")

    cases: List[Dict] = []
    summary = {
        "num_examples": 0,
        "local_correct": 0,
        "shadow_correct": 0,
        "fixed": 0,
        "broke": 0,
        "kept_correct": 0,
        "missed_wrong": 0,
        "teacher_calls": 0,
        "interventions": 0,
    }
    for row_id, gold_row in gold_rows.items():
        local = local_rows[row_id]
        shadow = shadow_rows[row_id]
        gold = str(gold_row.get("answer") or gold_row.get("gold") or gold_row.get("target") or gold_row.get("label") or "")
        local_ver = verify_text(local.get("output", ""), gold)
        shadow_ver = verify_text(shadow.get("output", ""), gold)
        local_correct = bool(local_ver.score)
        shadow_correct = bool(shadow_ver.score)
        outcome = _outcome(local_correct, shadow_correct)
        interventions = shadow.get("interventions", []) or []
        teacher_calls = int(shadow.get("teacher_calls", 0))
        case = {
            "id": row_id,
            "outcome": outcome,
            "prompt": gold_row.get("prompt") or gold_row.get("question") or gold_row.get("problem"),
            "gold": gold,
            "local_correct": local_correct,
            "shadow_correct": shadow_correct,
            "local_parsed_answer": local_ver.parsed_answer,
            "shadow_parsed_answer": shadow_ver.parsed_answer,
            "local_output": local.get("output", ""),
            "shadow_output": shadow.get("output", ""),
            "teacher_calls": teacher_calls,
            "interventions": interventions,
        }
        if "decision_trace" in shadow:
            case["decision_trace"] = shadow.get("decision_trace", [])
        cases.append(case)

        summary["num_examples"] += 1
        summary["local_correct"] += int(local_correct)
        summary["shadow_correct"] += int(shadow_correct)
        summary[outcome] += 1
        summary["teacher_calls"] += teacher_calls
        summary["interventions"] += len(interventions)

    n = max(1, summary["num_examples"])
    summary["local_accuracy"] = summary["local_correct"] / n
    summary["shadow_accuracy"] = summary["shadow_correct"] / n
    summary["net_gain"] = summary["fixed"] - summary["broke"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "cases_all.jsonl", cases)
    for name in ("fixed", "broke", "kept_correct", "missed_wrong"):
        write_jsonl(output_dir / f"{name}.jsonl", [case for case in cases if case["outcome"] == name])
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
