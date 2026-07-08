#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

from shadow_to_think.data_io import read_jsonl, write_jsonl


DEFAULT_POSITIVE_DECISIONS = ("accept_teacher", "accept_teacher_span")
DEFAULT_NEGATIVE_DECISIONS = ("keep_student", "select_student_topk", "abstain_no_verified_candidate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a relabeled ask-gate dataset from shadow collection records.")
    p.add_argument("--record_path", action="append", required=True, help="Shadow collection JSONL. Repeat for shards.")
    p.add_argument("--output_path", required=True)
    p.add_argument("--negative_per_positive", type=float, default=2.0)
    p.add_argument("--positive_decisions", default=",".join(DEFAULT_POSITIVE_DECISIONS))
    p.add_argument("--negative_decisions", default=",".join(DEFAULT_NEGATIVE_DECISIONS))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _split_csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    positive_decisions = _split_csv(args.positive_decisions)
    negative_decisions = _split_csv(args.negative_decisions)

    positives: List[Dict] = []
    negatives: List[Dict] = []
    decision_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for path in args.record_path:
        path_obj = Path(path)
        for rec in read_jsonl(path_obj):
            decision = str(rec.get("accept_decision", ""))
            decision_counts[decision] += 1
            examples = rec.get("trigger_examples", []) or []
            for idx, ex in enumerate(examples):
                if ex.get("features") is None:
                    continue
                label = None
                if decision in positive_decisions:
                    label = 1 if idx == 0 else 0
                elif decision in negative_decisions:
                    label = 0
                if label is None:
                    continue
                ex_out = dict(ex)
                ex_out["label"] = int(label)
                row = {
                    "id": f"{rec.get('id')}_{path_obj.name}_trigger{idx}",
                    "source_id": rec.get("id"),
                    "source_file": path_obj.name,
                    "accept_decision": decision,
                    "trigger_examples": [ex_out],
                }
                source_counts[path_obj.name] += 1
                if label:
                    positives.append(row)
                else:
                    negatives.append(row)

    if not positives:
        raise ValueError("No positive ask-gate examples found")
    random.shuffle(positives)
    random.shuffle(negatives)
    negative_limit = int(round(len(positives) * float(args.negative_per_positive)))
    rows = positives + negatives[:negative_limit]
    random.shuffle(rows)

    write_jsonl(args.output_path, rows)
    summary = {
        "output_path": args.output_path,
        "num_rows": len(rows),
        "num_positive": len(positives),
        "num_negative_pool": len(negatives),
        "num_negative_written": min(len(negatives), negative_limit),
        "negative_per_positive": args.negative_per_positive,
        "decision_counts": dict(decision_counts),
        "source_counts": dict(source_counts),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
