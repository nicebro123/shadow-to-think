#!/usr/bin/env python
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs" / "smoke" / "toy_shadow_v1.jsonl"
TRIGGER = ROOT / "runs" / "smoke" / "trigger.pt"
HIDDEN_SELECTOR = ROOT / "runs" / "smoke" / "hidden_selector.pt"


def make_toy_data(path: Path, n: int = 30, hidden_dim: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(13)
    rows = []
    for i in range(n):
        teacher_good = i % 2 == 0
        label_idx = 2 if teacher_good else 0
        cand_ids = [10, 11, 12]
        candidate_features = []
        for j in range(3):
            candidate_features.append([
                [-0.01, -0.20, -0.90][j] / 30.0,
                j / 3.0,
                1.0 if j == 0 else 0.0,
                1.0 if j == 0 else 0.0,
                1.0 if j == 2 else 0.0,
                0.8 if teacher_good else 0.2,
                0.4,
                0.1,
            ])
        prefix_hidden = [0.0] * hidden_dim
        prefix_hidden[0] = 1.0 if teacher_good else -1.0
        candidate_embeddings = []
        for j in range(3):
            emb = [0.0] * hidden_dim
            emb[0] = 1.0 if j == 2 else -1.0
            emb[1] = 1.0 if j == 0 else 0.0
            candidate_embeddings.append(emb)
        trigger_examples = []
        for j in range(4):
            is_pos = j == 1
            feats = [0.8 if is_pos else 0.1, 0.1 if is_pos else 0.8, -0.1, 0.3, j / 4, 0.2, 0.0, 0.0, 1.0 if is_pos else 0.0]
            trigger_examples.append({"features": feats, "label": 1 if is_pos else 0})
        rows.append({
            "id": f"toy_{i}",
            "candidate_ids": cand_ids,
            "candidate_texts": ["A", "B", "C"],
            "student_token_id": 10,
            "teacher_token_id": 12,
            "candidate_features": candidate_features,
            "label_idx": label_idx,
            "label_candidate_id": cand_ids[label_idx],
            "accept_decision": "accept_teacher" if label_idx == 2 else "keep_student",
            "prefix_hidden": prefix_hidden,
            "candidate_embeddings": candidate_embeddings,
            "trigger_examples": trigger_examples,
        })
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(cmd):
    print("$", " ".join(str(x) for x in cmd))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    subprocess.check_call(cmd, env=env)


def main() -> None:
    make_toy_data(DATA)
    run([sys.executable, str(ROOT / "scripts" / "train_trigger.py"), "--train_path", str(DATA), "--output_path", str(TRIGGER), "--epochs", "1", "--hidden_size", "16", "--lr", "0.01"])
    run([sys.executable, str(ROOT / "scripts" / "train_hidden_selector.py"), "--train_path", str(DATA), "--output_path", str(HIDDEN_SELECTOR), "--use_precomputed", "--epochs", "1", "--selector_dim", "16", "--lr", "0.01"])
    print("Shadow-to-Think v1 smoke test finished.")


if __name__ == "__main__":
    main()
