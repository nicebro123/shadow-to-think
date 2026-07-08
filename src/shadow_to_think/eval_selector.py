from __future__ import annotations

from typing import Dict, List

import torch

from .data_io import read_jsonl
from .selector_model import load_selector, selector_predict


def evaluate_selector(data_path: str, ckpt_path: str, device: str | None = None) -> Dict:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_selector(ckpt_path, map_location=device_obj).to(device_obj)
    model.eval()
    total = 0
    correct = 0
    accept_teacher_total = 0
    accept_teacher_correct = 0
    rows = []
    for rec in read_jsonl(data_path):
        if rec.get("label_idx") is None or not rec.get("candidate_features"):
            continue
        feats = torch.tensor(rec["candidate_features"], dtype=torch.float32, device=device_obj)
        pred_idx = selector_predict(model, feats)
        label_idx = int(rec["label_idx"])
        ok = pred_idx == label_idx
        total += 1
        correct += int(ok)
        candidate_ids = rec.get("candidate_ids", [])
        teacher_id = rec.get("teacher_token_id")
        pred_id = candidate_ids[pred_idx] if pred_idx < len(candidate_ids) else None
        if pred_id == teacher_id:
            accept_teacher_total += 1
            accept_teacher_correct += int(ok)
        rows.append({"id": rec.get("id"), "pred_idx": pred_idx, "label_idx": label_idx, "correct": ok})
    return {
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "accept_teacher_total": accept_teacher_total,
        "accept_teacher_precision": accept_teacher_correct / accept_teacher_total if accept_teacher_total else 0.0,
        "examples": rows[:20],
    }
