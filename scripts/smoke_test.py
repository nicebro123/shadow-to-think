#!/usr/bin/env python
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs" / "smoke" / "toy_shadow.jsonl"
CKPT = ROOT / "runs" / "smoke" / "selector.pt"


def make_toy_data(path: Path, n: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(7)
    rows = []
    for i in range(n):
        # Three candidates. The synthetic rule: if entropy high and teacher flag is on,
        # teacher candidate is usually the verified best; otherwise student top1 is best.
        entropy = random.random()
        teacher_good = entropy > 0.45
        candidate_features = []
        candidate_ids = [10, 11, 12]
        for j in range(3):
            is_top1 = 1.0 if j == 0 else 0.0
            is_student = 1.0 if j == 0 else 0.0
            is_teacher = 1.0 if j == 2 else 0.0
            logprob = [-0.05, -0.4, -0.8][j]
            rank = j / 3.0
            candidate_features.append([
                logprob / 30.0,
                rank,
                is_top1,
                is_student,
                is_teacher,
                entropy,
                0.5,
                0.2,
            ])
        label_idx = 2 if teacher_good else 0
        rows.append(
            {
                "id": f"toy_{i}",
                "candidate_ids": candidate_ids,
                "candidate_texts": ["A", "B", "C"],
                "student_token_id": 10,
                "teacher_token_id": 12,
                "candidate_features": candidate_features,
                "label_idx": label_idx,
                "label_candidate_id": candidate_ids[label_idx],
                "accept_decision": "accept_teacher" if label_idx == 2 else "keep_student",
            }
        )
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(cmd):
    print("$", " ".join(str(x) for x in cmd))
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(ROOT / "src") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    subprocess.check_call(cmd, env=env)


def main() -> None:
    make_toy_data(DATA)
    run([
        sys.executable,
        str(ROOT / "scripts" / "train_local_selector.py"),
        "--train_path", str(DATA),
        "--output_path", str(CKPT),
        "--epochs", "1",
        "--hidden_size", "32",
        "--lr", "0.01",
    ])
    run([
        sys.executable,
        str(ROOT / "scripts" / "eval_local_selector.py"),
        "--data_path", str(DATA),
        "--ckpt_path", str(CKPT),
    ])
    print("Smoke test finished.")


if __name__ == "__main__":
    main()
