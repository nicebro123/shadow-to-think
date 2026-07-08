from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.optim import AdamW
from tqdm import tqdm

from .data_io import read_jsonl
from .features import FEATURE_NAMES
from .generation import last_hidden_state
from .hidden_selector_model import HiddenStateSelector, hidden_selector_loss, hidden_selector_predict, save_hidden_selector
from .models import load_lm


def load_selector_records(path: str) -> List[Dict]:
    records: List[Dict] = []
    for rec in read_jsonl(path):
        if rec.get("label_idx") is None:
            continue
        if not rec.get("candidate_ids"):
            continue
        label_idx = int(rec["label_idx"])
        if 0 <= label_idx < len(rec["candidate_ids"]):
            records.append(rec)
    return records


def _tensor_features(rec: Dict, device) -> torch.Tensor:
    feats = rec.get("candidate_features")
    if feats is None:
        feats = [[0.0] * len(FEATURE_NAMES) for _ in rec["candidate_ids"]]
    return torch.tensor(feats, dtype=torch.float32, device=device)


@torch.no_grad()
def _record_tensors_from_lm(rec: Dict, student_model, tokenizer, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rec.get("prefix_at_div_ids"):
        ids = rec["prefix_at_div_ids"]
    else:
        text = rec.get("prefix_at_div_text") or rec.get("prefix_text") or rec.get("prompt", "")
        ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        ids = [getattr(tokenizer, "eos_token_id", 0)]
    prefix_ids = torch.tensor([ids], dtype=torch.long, device=device)
    h = last_hidden_state(student_model, prefix_ids).to(device)
    cand = torch.tensor(rec["candidate_ids"], dtype=torch.long, device=device)
    emb_layer = student_model.get_input_embeddings()
    embeds = emb_layer(cand).detach().float()
    feats = _tensor_features(rec, device)
    return h, embeds, feats


def _record_tensors_precomputed(rec: Dict, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h = torch.tensor(rec["prefix_hidden"], dtype=torch.float32, device=device)
    embeds = torch.tensor(rec["candidate_embeddings"], dtype=torch.float32, device=device)
    feats = _tensor_features(rec, device)
    return h, embeds, feats


def train_hidden_selector(
    train_path: str,
    output_path: str,
    *,
    student_model_name: str | None = None,
    use_precomputed: bool = False,
    selector_dim: int = 256,
    lr: float = 1e-3,
    epochs: int = 3,
    seed: int = 42,
    val_ratio: float = 0.1,
    device: str | None = None,
    dtype: str = "auto",
) -> Dict:
    random.seed(seed)
    torch.manual_seed(seed)
    device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device in (None, "auto") else torch.device(device)
    records = load_selector_records(train_path)
    if not records:
        raise ValueError("No trainable selector records found.")
    random.shuffle(records)
    val_n = max(1, int(len(records) * val_ratio)) if len(records) > 10 else 0
    val_records = records[:val_n]
    train_records = records[val_n:] if val_n else records

    loaded = None
    student_model = None
    tokenizer = None
    if not use_precomputed:
        if not student_model_name:
            raise ValueError("student_model_name is required unless --use_precomputed is set")
        loaded = load_lm(student_model_name, device=str(device_obj), dtype=dtype)
        student_model = loaded.model
        tokenizer = loaded.tokenizer
        student_model.eval()

    first = train_records[0]
    if use_precomputed:
        h0, e0, f0 = _record_tensors_precomputed(first, device_obj)
    else:
        h0, e0, f0 = _record_tensors_from_lm(first, student_model, tokenizer, device_obj)
    hidden_dim = int(h0.numel())
    feature_dim = int(f0.shape[-1])
    model = HiddenStateSelector(hidden_dim=hidden_dim, feature_dim=feature_dim, selector_dim=selector_dim).to(device_obj)
    opt = AdamW(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        random.shuffle(train_records)
        total_loss = 0.0
        for rec in tqdm(train_records, desc=f"hidden selector epoch {epoch+1}/{epochs}"):
            if use_precomputed:
                h, embeds, feats = _record_tensors_precomputed(rec, device_obj)
            else:
                h, embeds, feats = _record_tensors_from_lm(rec, student_model, tokenizer, device_obj)
            label = int(rec["label_idx"])
            loss = hidden_selector_loss(model, h, embeds, feats, label)
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
            with torch.no_grad():
                for rec in val_records:
                    if use_precomputed:
                        h, embeds, feats = _record_tensors_precomputed(rec, device_obj)
                    else:
                        h, embeds, feats = _record_tensors_from_lm(rec, student_model, tokenizer, device_obj)
                    pred = hidden_selector_predict(model, h, embeds, feats)
                    correct += int(pred == int(rec["label_idx"]))
            val_acc = correct / len(val_records)
            history["val_acc"].append(val_acc)
            print(f"epoch={epoch+1} loss={avg_loss:.4f} val_acc={val_acc:.4f}")
        else:
            print(f"epoch={epoch+1} loss={avg_loss:.4f}")

    save_hidden_selector(
        model,
        output_path,
        extra={"feature_names": FEATURE_NAMES, "history": history, "num_records": len(records), "use_precomputed": use_precomputed},
    )
    return {"output_path": str(output_path), "num_records": len(records), "history": history}
