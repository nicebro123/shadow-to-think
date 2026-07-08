#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from shadow_to_think.data_io import infer_prompt, read_jsonl, write_jsonl
from shadow_to_think.decode_controller import DecodeConfig, ShadowDecodeController
from shadow_to_think.hidden_selector_model import load_hidden_selector
from shadow_to_think.models import load_lm
from shadow_to_think.teacher_backend import TransformersTeacherBackend, VLLMTeacherClient
from shadow_to_think.trigger_model import load_trigger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run generation-time Shadow-to-Think decoding.")
    p.add_argument("--student_model", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--mode", choices=["shadow", "local"], default="shadow")
    p.add_argument("--teacher_backend", choices=["transformers", "vllm"], default="transformers")
    p.add_argument("--teacher_model", default=None)
    p.add_argument("--teacher_base_url", default="http://localhost:8000/v1")
    p.add_argument("--teacher_api_key", default="EMPTY")
    p.add_argument("--teacher_model_name", default=None)
    p.add_argument("--trigger_ckpt", default=None)
    p.add_argument("--selector_ckpt", default=None)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--draft_len", type=int, default=32)
    p.add_argument("--shadow_len", type=int, default=32)
    p.add_argument("--student_topk", type=int, default=16)
    p.add_argument("--trigger_threshold", type=float, default=0.5)
    p.add_argument("--teacher_span_len", type=int, default=1)
    p.add_argument("--teacher_span_mode", choices=["fixed", "step"], default="fixed")
    p.add_argument("--teacher_span_min_len", type=int, default=4)
    p.add_argument("--skip_style_divergence", action="store_true")
    p.add_argument("--min_divergence_index", type=int, default=0)
    p.add_argument("--intervention_policy", choices=["selector", "teacher_only"], default="selector")
    p.add_argument("--require_math_signal_divergence", action="store_true")
    p.add_argument("--math_signal_window", type=int, default=8)
    p.add_argument("--max_teacher_calls", type=int, default=None)
    p.add_argument("--trace_decisions", action="store_true", help="Write trigger scores/features for later protection training.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    student = load_lm(args.student_model, device=args.device, dtype=args.dtype)
    teacher_backend = None
    if args.mode == "shadow":
        if args.teacher_backend == "transformers":
            if not args.teacher_model:
                raise ValueError("--teacher_model required for transformers teacher backend")
            teacher = load_lm(args.teacher_model, device=args.device, dtype=args.dtype)
            teacher_backend = TransformersTeacherBackend(teacher.model)
        else:
            teacher_backend = VLLMTeacherClient(
                base_url=args.teacher_base_url,
                model=args.teacher_model_name or args.teacher_model or args.student_model,
                api_key=args.teacher_api_key,
            )
    trigger = load_trigger(args.trigger_ckpt, map_location=student.device) if args.trigger_ckpt else None
    selector = load_hidden_selector(args.selector_ckpt, map_location=student.device) if args.selector_ckpt else None
    config = DecodeConfig(
        max_new_tokens=args.max_new_tokens,
        draft_len=args.draft_len,
        shadow_len=args.shadow_len,
        student_topk=args.student_topk,
        temperature=args.temperature,
        trigger_threshold=args.trigger_threshold,
        mode=args.mode,
        teacher_span_len=args.teacher_span_len,
        teacher_span_mode=args.teacher_span_mode,
        teacher_span_min_len=args.teacher_span_min_len,
        skip_style_divergence=args.skip_style_divergence,
        min_divergence_index=args.min_divergence_index,
        intervention_policy=args.intervention_policy,
        require_math_signal_divergence=args.require_math_signal_divergence,
        math_signal_window=args.math_signal_window,
        max_teacher_calls=args.max_teacher_calls,
        trace_decisions=args.trace_decisions,
    )
    controller = ShadowDecodeController(
        student.model,
        student.tokenizer,
        device=student.device,
        teacher_backend=teacher_backend,
        trigger=trigger,
        hidden_selector=selector,
        config=config,
    )
    rows = []
    for i, row in enumerate(tqdm(read_jsonl(args.dataset_path), desc="decode")):
        if args.max_samples is not None and i >= args.max_samples:
            break
        prompt = infer_prompt(row)
        out = controller.generate(prompt)
        record = {
            "id": row.get("id"),
            "prompt": prompt,
            "output": out["text"],
            "interventions": out["interventions"],
            "teacher_calls": out["teacher_calls"],
        }
        if args.trace_decisions:
            record["decision_trace"] = out.get("decision_trace", [])
        rows.append(record)
    write_jsonl(args.output_path, rows, append=False)
    summary = {
        "num_examples": len(rows),
        "total_teacher_calls": sum(r["teacher_calls"] for r in rows),
        "total_interventions": sum(len(r["interventions"]) for r in rows),
        "output_path": args.output_path,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
