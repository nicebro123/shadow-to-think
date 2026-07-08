#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from shadow_to_think.train_trigger import train_trigger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the student-side local trigger.")
    p.add_argument("--train_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_trigger(
        args.train_path,
        args.output_path,
        hidden_size=args.hidden_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        val_ratio=args.val_ratio,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
