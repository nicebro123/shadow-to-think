#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from shadow_to_think.train_hidden_selector import train_hidden_selector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train hidden-state candidate selector.")
    p.add_argument("--train_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--student_model", default=None, help="Required unless --use_precomputed is set.")
    p.add_argument("--use_precomputed", action="store_true", help="Use prefix_hidden and candidate_embeddings from JSONL.")
    p.add_argument("--selector_dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_hidden_selector(
        args.train_path,
        args.output_path,
        student_model_name=args.student_model,
        use_precomputed=args.use_precomputed,
        selector_dim=args.selector_dim,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        val_ratio=args.val_ratio,
        device=args.device,
        dtype=args.dtype,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
