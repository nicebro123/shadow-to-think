#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from shadow_to_think.data_io import read_jsonl
from shadow_to_think.models import load_lm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optional: directly distill verified next-token corrections into the student LM.")
    p.add_argument("--student_model", required=True)
    p.add_argument("--train_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_prefix_tokens", type=int, default=2048)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_records(path: str, max_samples: int | None = None):
    rows = []
    for rec in read_jsonl(path):
        if rec.get("label_candidate_id") is None:
            continue
        prefix_ids = rec.get("prefix_at_div_ids")
        prefix_text = rec.get("prefix_at_div_text") or rec.get("prefix_text")
        if not prefix_ids and not prefix_text:
            continue
        rows.append({"prefix_ids": prefix_ids, "prefix": prefix_text, "label_id": int(rec["label_candidate_id"])})
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def maybe_apply_lora(model, args):
    if not args.use_lora:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("Install peft or remove --use_lora") from exc
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    loaded = load_lm(args.student_model, device=args.device, dtype=args.dtype)
    model = maybe_apply_lora(loaded.model, args)
    tokenizer = loaded.tokenizer
    device = loaded.device
    model.train()

    # If not using LoRA, this fine-tunes all parameters. For real runs, prefer --use_lora.
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=args.lr)
    rows = load_records(args.train_path, args.max_samples)
    if not rows:
        raise ValueError("No training records found with prefix_at_div_text and label_candidate_id")

    for epoch in range(args.epochs):
        random.shuffle(rows)
        total_loss = 0.0
        for row in tqdm(rows, desc=f"epoch {epoch+1}/{args.epochs}"):
            if row.get("prefix_ids"):
                ids = [int(x) for x in row["prefix_ids"]]
            else:
                ids = tokenizer.encode(row["prefix"], add_special_tokens=False)
            ids = ids[-args.max_prefix_tokens:]
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            label = torch.tensor([row["label_id"]], dtype=torch.long, device=device)
            out = model(input_ids)
            logits = out.logits[:, -1, :]
            loss = F.cross_entropy(logits, label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            total_loss += float(loss.item())
        print(f"epoch={epoch+1} loss={total_loss / max(1, len(rows)):.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.use_lora and hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    print(f"Saved trained student/adapter to {output_dir}")


if __name__ == "__main__":
    main()
