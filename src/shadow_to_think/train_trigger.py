from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import AdamW
from tqdm import tqdm

from .data_io import read_jsonl
from .features import TRIGGER_FEATURE_NAMES
from .trigger_model import FeatureTrigger, save_trigger, trigger_loss, trigger_score


def load_trigger_examples(path: str) -> List[Dict]:
    examples: List[Dict] = []
    for rec in read_jsonl(path):
        for ex in rec.get("trigger_examples", []) or []:
            feats = ex.get("features")
            if feats is None:
                continue
            examples.append({"features": feats, "label": int(ex.get("label", 0)), "source_id": rec.get("id")})
    return examples


def train_trigger(
    train_path: str,
    output_path: str,
    *,
    hidden_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 3,
    seed: int = 42,
    val_ratio: float = 0.1,
    device: str | None = None,
) -> Dict:
    random.seed(seed)
    torch.manual_seed(seed)
    device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device in (None, "auto") else torch.device(device)
    examples = load_trigger_examples(train_path)
    if not examples:
        raise ValueError("No trigger_examples found. Re-run collection with the v1 collector.")
    random.shuffle(examples)
    val_n = max(1, int(len(examples) * val_ratio)) if len(examples) > 10 else 0
    val_examples = examples[:val_n]
    train_examples = examples[val_n:] if val_n else examples
    feature_dim = len(train_examples[0]["features"])
    model = FeatureTrigger(feature_dim=feature_dim, hidden_size=hidden_size).to(device_obj)
    opt = AdamW(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_acc": [], "val_pos_recall": []}
    for epoch in range(epochs):
        model.train()
        random.shuffle(train_examples)
        total_loss = 0.0
        for ex in tqdm(train_examples, desc=f"trigger epoch {epoch+1}/{epochs}"):
            feats = torch.tensor(ex["features"], dtype=torch.float32, device=device_obj)
            label = torch.tensor([float(ex["label"])], dtype=torch.float32, device=device_obj)
            loss = trigger_loss(model, feats.unsqueeze(0), label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, len(train_examples))
        history["train_loss"].append(avg_loss)
        if val_examples:
            model.eval()
            correct = 0
            pos_total = 0
            pos_hit = 0
            for ex in val_examples:
                feats = torch.tensor(ex["features"], dtype=torch.float32, device=device_obj)
                score = trigger_score(model, feats)
                pred = 1 if score >= 0.5 else 0
                label = int(ex["label"])
                correct += int(pred == label)
                if label == 1:
                    pos_total += 1
                    pos_hit += int(pred == 1)
            history["val_acc"].append(correct / len(val_examples))
            history["val_pos_recall"].append(pos_hit / pos_total if pos_total else 0.0)
            print(f"epoch={epoch+1} loss={avg_loss:.4f} val_acc={history['val_acc'][-1]:.4f} pos_recall={history['val_pos_recall'][-1]:.4f}")
        else:
            print(f"epoch={epoch+1} loss={avg_loss:.4f}")
    save_trigger(model, output_path, extra={"feature_names": TRIGGER_FEATURE_NAMES, "history": history, "num_examples": len(examples)})
    return {"output_path": str(output_path), "num_examples": len(examples), "history": history}
