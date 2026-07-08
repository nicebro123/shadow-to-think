#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from tqdm import tqdm

from shadow_to_think.data_io import infer_gold, infer_prompt, read_jsonl, write_jsonl
from shadow_to_think.generation import encode_prompt, greedy_generate_new_ids
from shadow_to_think.models import load_lm
from shadow_to_think.verifier import verify_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a student LoRA adapter with local decoding.")
    p.add_argument("--student_model", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_prompt_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from peft import PeftModel

    loaded = load_lm(args.student_model, device=args.device, dtype=args.dtype)
    model = PeftModel.from_pretrained(loaded.model, args.adapter_path)
    model.eval()
    tokenizer = loaded.tokenizer
    rows = []
    for i, row in enumerate(tqdm(read_jsonl(args.dataset_path), desc="eval lora")):
        if args.max_samples is not None and i >= args.max_samples:
            break
        prompt = infer_prompt(row)
        input_ids = encode_prompt(tokenizer, prompt, args.max_prompt_tokens, device=loaded.device)
        generated = greedy_generate_new_ids(
            model,
            input_ids,
            args.max_new_tokens,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            temperature=args.temperature,
        )
        output = tokenizer.decode(generated, skip_special_tokens=True)
        gold = infer_gold(row)
        verification = verify_text(output, gold)
        rows.append(
            {
                "id": row.get("id"),
                "prompt": prompt,
                "gold": gold,
                "output": output,
                "parsed_answer": verification.parsed_answer,
                "score": verification.score,
            }
        )
    write_jsonl(args.output_path, rows, append=False)
    total = len(rows)
    correct = int(sum(r["score"] for r in rows))
    print(
        json.dumps(
            {
                "num_examples": total,
                "correct": correct,
                "accuracy": correct / total if total else 0.0,
                "output_path": args.output_path,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
