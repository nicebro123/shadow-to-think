#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

from shadow_to_think.data_io import read_jsonl, write_jsonl


DEFAULT_LABELS = {
    "fixed": 1,
    "broke": 0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a gate dataset from traced decode decisions and final outcomes.")
    p.add_argument("--cases_path", required=True, help="cases_all.jsonl from analyze_shadow_outcomes.py")
    p.add_argument("--output_path", required=True)
    p.add_argument("--negative_per_positive", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    positives: List[Dict] = []
    negatives: List[Dict] = []
    outcome_counts: Counter[str] = Counter()
    traced_counts: Counter[str] = Counter()

    for case in read_jsonl(args.cases_path):
        outcome = str(case.get("outcome", ""))
        outcome_counts[outcome] += 1
        if outcome not in DEFAULT_LABELS:
            continue
        label = DEFAULT_LABELS[outcome]
        traces = case.get("decision_trace", []) or []
        for idx, trace in enumerate(traces):
            if trace.get("action") != "call_teacher":
                continue
            features = trace.get("trigger_features")
            if features is None:
                continue
            traced_counts[outcome] += 1
            row = {
                "id": f"{case.get('id')}_trace{idx}",
                "source_id": case.get("id"),
                "outcome": outcome,
                "teacher_calls": case.get("teacher_calls", 0),
                "trigger_score": trace.get("trigger_score"),
                "trigger_examples": [
                    {
                        "position_index": trace.get("position_index", 0),
                        "label": label,
                        "features": features,
                        "entropy": trace.get("entropy"),
                        "top1_margin": trace.get("top1_margin"),
                        "next_token_id": trace.get("top1_id"),
                        "next_token_text": trace.get("top1_text"),
                    }
                ],
            }
            if label:
                positives.append(row)
            else:
                negatives.append(row)

    if not positives:
        raise ValueError("No positive traced gate examples found; rerun decode with --trace_decisions")
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
        "outcome_counts": dict(outcome_counts),
        "traced_counts": dict(traced_counts),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
