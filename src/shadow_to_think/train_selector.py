from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
torch.set_num_threads(1)
from torch.optim import AdamW
from tqdm import tqdm

from .data_io import read_jsonl
from .features import FEATURE_NAMES
from .selector_model import FeatureSelector, save_selector, selector_loss, selector_predict


def load_trainable_records(path: str) -> List[Dict]:
    records = []
    for rec in read_jsonl(path):
        if rec.get("label_idx") is None:
            continue
        feats = rec.get("candidate_features")
        if not feats:
            continue
        label_idx = int(rec["label_idx"])
        if 0 <= label_idx < len(feats):
            records.append(rec)
    return records


def train_feature_selector(
    train_path: str,
    output_path: str,
    *,
    hidden_size: int = 128,
    lr: float = 1e-3,
    epochs: int = 3,
    seed: int = 42,
    val_ratio: float = 0.1,
    device: str | None = None,
) -> Dict:
    random.seed(seed)
    torch.manual_seed(seed)
    device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device in (None, "auto") else torch.device(device)
    records = load_trainable_records(train_path)
    if not records:
        raise ValueError("No trainable records found. Need JSONL records with candidate_features and label_idx.")
    random.shuffle(records)
    val_n = max(1, int(len(records) * val_ratio)) if len(records) > 5 else 0
    val_records = records[:val_n]
    train_records = records[val_n:] if val_n else records

    feature_dim = len(train_records[0]["candidate_features"][0])
    model = FeatureSelector(feature_dim=feature_dim, hidden_size=hidden_size).to(device_obj)
    opt = AdamW(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        random.shuffle(train_records)
        total_loss = 0.0
        for rec in tqdm(train_records, desc=f"epoch {epoch+1}/{epochs}"):
            feats = torch.tensor(rec["candidate_features"], dtype=torch.float32, device=device_obj)
            label = int(rec["label_idx"])
            loss = selector_loss(model, feats, label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, len(train_records))
        history["train_loss"].append(avg_loss)

        if val_records:
            model.eval()
            correct = 0
            for rec in val_records:
                feats = torch.tensor(rec["candidate_features"], dtype=torch.float32, device=device_obj)
                pred = selector_predict(model, feats)
                correct += int(pred == int(rec["label_idx"]))
            val_acc = correct / len(val_records)
            history["val_acc"].append(val_acc)
            print(f"epoch={epoch+1} train_loss={avg_loss:.4f} val_acc={val_acc:.4f}")
        else:
            print(f"epoch={epoch+1} train_loss={avg_loss:.4f}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_selector(
        model,
        str(output),
        extra={"feature_names": FEATURE_NAMES, "history": history, "num_records": len(records)},
    )
    return {"output_path": str(output), "history": history, "num_records": len(records)}
