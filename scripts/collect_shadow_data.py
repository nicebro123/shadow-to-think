#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from shadow_to_think.collector import collect_one_shadow_record, collect_one_step_shadow_record
from shadow_to_think.data_io import read_jsonl, write_jsonl
from shadow_to_think.models import ensure_same_tokenizer, load_lm
from shadow_to_think.teacher_backend import VLLMTeacherClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect Shadow-to-Think correction data.")
    p.add_argument("--student_model", required=True)
    p.add_argument("--teacher_model", required=True)
    p.add_argument("--dataset_path", required=True, help="JSONL with prompt/question and optional answer/gold.")
    p.add_argument("--output_path", required=True)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_prompt_tokens", type=int, default=2048)
    p.add_argument("--draft_len", type=int, default=32)
    p.add_argument("--shadow_len", type=int, default=32)
    p.add_argument("--student_topk", type=int, default=16)
    p.add_argument("--rollout_len", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--allow_no_gold", action="store_true")
    p.add_argument("--min_accept_score", type=float, default=0.5)
    p.add_argument("--scan_divergences", type=int, default=1)
    p.add_argument("--min_divergence_index", type=int, default=0)
    p.add_argument("--skip_style_divergence", action="store_true")
    p.add_argument("--teacher_span_len", type=int, default=1)
    p.add_argument("--warm_start_len", type=int, default=0)
    p.add_argument("--require_math_signal_divergence", action="store_true")
    p.add_argument("--math_signal_window", type=int, default=8)
    p.add_argument("--append", action="store_true", help="Append to output JSONL instead of overwriting.")
    p.add_argument("--teacher_backend", choices=["transformers", "vllm"], default="transformers")
    p.add_argument("--teacher_base_url", default="http://localhost:8000/v1")
    p.add_argument("--teacher_api_key", default="EMPTY")
    p.add_argument("--teacher_model_name", default=None, help="Model name served by vLLM; defaults to --teacher_model.")
    p.add_argument("--skip_tokenizer_check", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    student = load_lm(args.student_model, device=args.device, dtype=args.dtype)
    teacher = None
    teacher_backend = None
    if args.teacher_backend == "transformers":
        teacher = load_lm(args.teacher_model, device=args.device, dtype=args.dtype)
        if not args.skip_tokenizer_check:
            ensure_same_tokenizer(student.tokenizer, teacher.tokenizer)
    else:
        teacher_backend = VLLMTeacherClient(
            base_url=args.teacher_base_url,
            model=args.teacher_model_name or args.teacher_model,
            api_key=args.teacher_api_key,
        )

    rows = read_jsonl(args.dataset_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    buffer = []
    use_step_collector = (
        args.scan_divergences > 1
        or args.min_divergence_index > 0
        or args.skip_style_divergence
        or args.teacher_span_len > 1
        or args.warm_start_len > 0
        or args.require_math_signal_divergence
    )
    for i, row in enumerate(tqdm(rows, desc="collect")):
        if args.max_samples is not None and i >= args.max_samples:
            break
        common_kwargs = dict(
            teacher_backend=teacher_backend,
            device=student.device,
            max_prompt_tokens=args.max_prompt_tokens,
            draft_len=args.draft_len,
            shadow_len=args.shadow_len,
            student_topk=args.student_topk,
            rollout_len=args.rollout_len,
            temperature=args.temperature,
            allow_no_gold=args.allow_no_gold,
            min_accept_score=args.min_accept_score,
        )
        if use_step_collector:
            rec = collect_one_step_shadow_record(
                row,
                student.model,
                teacher.model if teacher is not None else None,
                student.tokenizer,
                scan_divergences=args.scan_divergences,
                min_divergence_index=args.min_divergence_index,
                skip_style_divergence=args.skip_style_divergence,
                teacher_span_len=args.teacher_span_len,
                warm_start_len=args.warm_start_len,
                require_math_signal_divergence=args.require_math_signal_divergence,
                math_signal_window=args.math_signal_window,
                **common_kwargs,
            )
        else:
            rec = collect_one_shadow_record(
                row,
                student.model,
                teacher.model if teacher is not None else None,
                student.tokenizer,
                **common_kwargs,
            )
        if rec is not None:
            buffer.append(rec)
            written += 1
        if len(buffer) >= 16:
            write_jsonl(output_path, buffer, append=args.append or (written > len(buffer)))
            buffer = []
    if buffer:
        write_jsonl(output_path, buffer, append=args.append or (written > len(buffer)))
    print(f"Wrote {written} shadow records to {output_path}")


if __name__ == "__main__":
    main()
