#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from shadow_to_think.data_io import read_jsonl
from shadow_to_think.models import load_lm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill verified step spans into the student with LoRA.")
    p.add_argument("--student_model", required=True)
    p.add_argument("--train_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_prefix_tokens", type=int, default=1024)
    p.add_argument("--max_target_tokens", type=int, default=64)
    p.add_argument("--teacher_target_upsample", type=int, default=4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    return p.parse_args()


def load_span_records(path: str, max_samples: int | None, teacher_target_upsample: int):
    rows = []
    for rec in read_jsonl(path):
        label_idx = rec.get("label_idx")
        if label_idx is None:
            continue
        prefix_ids = rec.get("prefix_at_div_ids")
        if not prefix_ids:
            continue
        target_ids = None
        seeds = rec.get("candidate_seed_ids")
        if isinstance(seeds, list) and int(label_idx) < len(seeds):
            target_ids = seeds[int(label_idx)]
        elif rec.get("label_candidate_id") is not None:
            target_ids = [int(rec["label_candidate_id"])]
        if not target_ids:
            continue
        row = {
            "id": rec.get("id"),
            "prefix_ids": [int(x) for x in prefix_ids],
            "target_ids": [int(x) for x in target_ids],
            "decision": rec.get("accept_decision"),
        }
        repeat = 1
        if str(rec.get("accept_decision", "")).startswith("accept_teacher"):
            repeat = max(1, int(teacher_target_upsample))
        rows.extend([row] * repeat)
        if max_samples is not None and len(rows) >= max_samples:
            rows = rows[:max_samples]
            break
    return rows


def apply_lora(model, args):
    from peft import LoraConfig, get_peft_model

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
    model = apply_lora(loaded.model, args)
    tokenizer = loaded.tokenizer
    device = loaded.device
    model.train()
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    rows = load_span_records(args.train_path, args.max_samples, args.teacher_target_upsample)
    if not rows:
        raise ValueError("No labeled span records found")
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    for epoch in range(args.epochs):
        random.shuffle(rows)
        total_loss = 0.0
        for row in tqdm(rows, desc=f"span lora epoch {epoch + 1}/{args.epochs}"):
            prefix = row["prefix_ids"][-args.max_prefix_tokens :]
            target = row["target_ids"][: args.max_target_tokens]
            input_ids = torch.tensor([prefix + target], dtype=torch.long, device=device)
            labels = torch.tensor([[-100] * len(prefix) + target], dtype=torch.long, device=device)
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            total_loss += float(loss.item())
        print(f"epoch={epoch + 1} loss={total_loss / max(1, len(rows)):.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
