#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from shadow_to_think.eval_selector import evaluate_selector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the MVP local selector on collected shadow data.")
    p.add_argument("--data_path", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_selector(args.data_path, args.ckpt_path, device=args.device)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
