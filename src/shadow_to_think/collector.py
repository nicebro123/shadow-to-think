from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .data_io import infer_gold, infer_prompt
from .divergence import find_first_meaningful_divergence, find_meaningful_divergences
from .features import build_candidate_features, build_trigger_features
from .generation import encode_prompt, greedy_generate_new_ids, next_token_stats
from .teacher_backend import TeacherBackend, TransformersTeacherBackend
from .verifier import verify_text


def _append_unique(base: List[int], extra: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in list(base) + list(extra):
        x = int(x)
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


@torch.no_grad()
def _trigger_example_for_position(
    student_model,
    tokenizer,
    device,
    prefix_base_ids: List[int],
    draft_prefix: List[int],
    position_index: int,
    draft_len: int,
    label: int,
    topk: int,
) -> Dict[str, Any]:
    pos_prefix = torch.tensor([prefix_base_ids + draft_prefix[:position_index]], dtype=torch.long, device=device)
    stats = next_token_stats(student_model, pos_prefix, topk=topk)
    next_tok = draft_prefix[position_index] if position_index < len(draft_prefix) else stats["top1_id"]
    return {
        "position_index": int(position_index),
        "label": int(label),
        "next_token_id": int(next_tok),
        "next_token_text": tokenizer.decode([int(next_tok)], skip_special_tokens=False),
        "features": build_trigger_features(
            stats=stats,
            position_index=position_index,
            draft_len=draft_len,
            next_token_id=next_tok,
            tokenizer=tokenizer,
        ),
        "entropy": float(stats["entropy"]),
        "top1_margin": float(stats["top1_margin"]),
    }


@torch.no_grad()
def collect_one_shadow_record(
    row: Dict[str, Any],
    student_model,
    teacher_model=None,
    tokenizer=None,
    *,
    teacher_backend: TeacherBackend | None = None,
    device,
    max_prompt_tokens: int = 2048,
    draft_len: int = 32,
    shadow_len: int = 32,
    student_topk: int = 16,
    rollout_len: int = 96,
    temperature: float = 0.0,
    allow_no_gold: bool = False,
    min_accept_score: float = 0.5,
    trigger_negative_window: int = 3,
) -> Optional[Dict[str, Any]]:
    """Collect one Shadow-to-Think correction sample.

    The teacher may be passed either as a local HF model (`teacher_model`) or as
    a `teacher_backend` such as the vLLM OpenAI-compatible client.
    """
    if tokenizer is None:
        raise ValueError("tokenizer must be provided")
    prompt = infer_prompt(row)
    gold = infer_gold(row)
    if gold is None and not allow_no_gold:
        return None

    prefix_ids = encode_prompt(tokenizer, prompt, max_prompt_tokens=max_prompt_tokens, device=device)
    prefix_ids_list = prefix_ids.squeeze(0).tolist()
    eos_id = getattr(tokenizer, "eos_token_id", None)

    if teacher_backend is None:
        if teacher_model is None:
            raise ValueError("Need either teacher_model or teacher_backend")
        teacher_backend = TransformersTeacherBackend(teacher_model)

    student_draft = greedy_generate_new_ids(
        student_model, prefix_ids, draft_len, eos_token_id=eos_id, temperature=temperature
    )
    teacher_shadow = teacher_backend.generate_ids(tokenizer, prefix_ids, shadow_len, temperature=temperature)

    div = find_first_meaningful_divergence(student_draft, teacher_shadow, tokenizer)
    if div is None:
        return None

    prefix_at_div_ids_list = prefix_ids_list + student_draft[: div.index]
    prefix_at_div_ids = torch.tensor([prefix_at_div_ids_list], dtype=torch.long, device=device)
    stats = next_token_stats(student_model, prefix_at_div_ids, topk=student_topk)
    topk_ids = stats["topk_ids"]
    topk_logprobs = stats["topk_logprobs"]
    entropy = stats["entropy"]

    candidate_ids = _append_unique(topk_ids, [div.teacher_token_id])

    rollout_results: List[Dict[str, Any]] = []
    best_idx: Optional[int] = None
    best_score = -1.0
    for idx, cid in enumerate(candidate_ids):
        rollout_prefix = torch.tensor([prefix_at_div_ids_list + [int(cid)]], dtype=torch.long, device=device)
        rollout_new = greedy_generate_new_ids(
            student_model, rollout_prefix, rollout_len, eos_token_id=eos_id, temperature=temperature
        )
        # Important: verify only the candidate-conditioned generated suffix, not
        # the original prompt. Otherwise numbers in the prompt can cause false positives.
        generated_ids = [int(cid)] + rollout_new
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_text = tokenizer.decode(rollout_prefix.squeeze(0).tolist() + rollout_new, skip_special_tokens=True)
        verification = verify_text(generated_text, gold)
        score = verification.score
        rollout_results.append(
            {
                "candidate_id": int(cid),
                "candidate_text": tokenizer.decode([int(cid)], skip_special_tokens=False),
                "generated_text": generated_text,
                "rollout_text": full_text,
                "parsed_answer": verification.parsed_answer,
                "score": score,
            }
        )
        if score > best_score:
            best_score = score
            best_idx = idx

    if gold is None:
        label_idx = None
        decision = "unlabeled"
    elif best_idx is None or best_score < float(min_accept_score):
        # Do not fabricate selector labels when every candidate fails.
        label_idx = None
        decision = "abstain_no_verified_candidate"
    else:
        label_idx = int(best_idx)
        best_candidate = int(candidate_ids[label_idx])
        if best_candidate == int(div.teacher_token_id) and best_candidate != int(div.student_token_id):
            decision = "accept_teacher"
        elif best_candidate == int(div.student_token_id):
            decision = "keep_student"
        else:
            decision = "select_student_topk"

    candidate_features = build_candidate_features(
        candidate_ids=candidate_ids,
        student_topk_ids=topk_ids,
        student_topk_logprobs=topk_logprobs,
        student_token_id=div.student_token_id,
        teacher_token_id=div.teacher_token_id,
        entropy=entropy,
        divergence_index=div.index,
        draft_len=draft_len,
        tokenizer=tokenizer,
    )

    trigger_examples: List[Dict[str, Any]] = []
    # Positive example: first meaningful divergence position.
    trigger_examples.append(
        _trigger_example_for_position(
            student_model,
            tokenizer,
            device,
            prefix_ids_list,
            student_draft,
            div.index,
            draft_len,
            label=1,
            topk=student_topk,
        )
    )
    # Nearby negatives: cheap local trigger supervision for student-side risky-position detection.
    neg_positions = []
    for j in range(max(0, div.index - trigger_negative_window), min(len(student_draft), div.index + trigger_negative_window + 1)):
        if j != div.index:
            neg_positions.append(j)
    if 0 not in neg_positions and div.index != 0 and len(student_draft) > 0:
        neg_positions.append(0)
    for j in neg_positions:
        trigger_examples.append(
            _trigger_example_for_position(
                student_model,
                tokenizer,
                device,
                prefix_ids_list,
                student_draft,
                j,
                draft_len,
                label=0,
                topk=student_topk,
            )
        )

    return {
        "id": row.get("id"),
        "prompt": prompt,
        "gold": gold,
        "prefix_ids": prefix_ids_list,
        "prefix_text": tokenizer.decode(prefix_ids_list, skip_special_tokens=True),
        "prefix_at_div_ids": prefix_at_div_ids_list,
        "prefix_at_div_text": tokenizer.decode(prefix_at_div_ids_list, skip_special_tokens=True),
        "student_draft_ids": student_draft,
        "teacher_shadow_ids": teacher_shadow,
        "student_draft_text": tokenizer.decode(student_draft, skip_special_tokens=False),
        "teacher_shadow_text": tokenizer.decode(teacher_shadow, skip_special_tokens=False),
        "divergence_index": div.index,
        "divergence_reason": div.reason,
        "student_token_id": int(div.student_token_id),
        "teacher_token_id": int(div.teacher_token_id),
        "student_token_text": div.student_text,
        "teacher_token_text": div.teacher_text,
        "student_topk_ids": topk_ids,
        "student_topk_logprobs": topk_logprobs,
        "student_entropy": entropy,
        "student_top1_margin": float(stats.get("top1_margin", 0.0)),
        "candidate_ids": candidate_ids,
        "candidate_texts": [tokenizer.decode([int(x)], skip_special_tokens=False) for x in candidate_ids],
        "candidate_features": candidate_features,
        "label_idx": label_idx,
        "label_candidate_id": int(candidate_ids[label_idx]) if label_idx is not None else None,
        "best_score": best_score,
        "accept_decision": decision,
        "rollout_results": rollout_results,
        "trigger_examples": trigger_examples,
    }


@torch.no_grad()
def collect_one_step_shadow_record(
    row: Dict[str, Any],
    student_model,
    teacher_model=None,
    tokenizer=None,
    *,
    teacher_backend: TeacherBackend | None = None,
    device,
    max_prompt_tokens: int = 2048,
    draft_len: int = 32,
    shadow_len: int = 32,
    student_topk: int = 16,
    rollout_len: int = 96,
    temperature: float = 0.0,
    allow_no_gold: bool = False,
    min_accept_score: float = 0.5,
    trigger_negative_window: int = 3,
    scan_divergences: int = 8,
    min_divergence_index: int = 0,
    skip_style_divergence: bool = True,
    teacher_span_len: int = 8,
    warm_start_len: int = 0,
    require_math_signal_divergence: bool = False,
    math_signal_window: int = 8,
) -> Optional[Dict[str, Any]]:
    """Collect a math-oriented record by scanning later divergences.

    This keeps the selector target token-level, but scores a teacher candidate
    by letting it carry a short teacher span before handing control back to the
    student. That avoids learning from harmless opening-style divergences.
    """
    if tokenizer is None:
        raise ValueError("tokenizer must be provided")
    prompt = infer_prompt(row)
    gold = infer_gold(row)
    if gold is None and not allow_no_gold:
        return None

    prefix_ids = encode_prompt(tokenizer, prompt, max_prompt_tokens=max_prompt_tokens, device=device)
    prefix_ids_list = prefix_ids.squeeze(0).tolist()
    eos_id = getattr(tokenizer, "eos_token_id", None)

    if teacher_backend is None:
        if teacher_model is None:
            raise ValueError("Need either teacher_model or teacher_backend")
        teacher_backend = TransformersTeacherBackend(teacher_model)

    student_draft = greedy_generate_new_ids(
        student_model, prefix_ids, draft_len, eos_token_id=eos_id, temperature=temperature
    )
    teacher_shadow = teacher_backend.generate_ids(tokenizer, prefix_ids, shadow_len, temperature=temperature)
    divs = find_meaningful_divergences(
        student_draft,
        teacher_shadow,
        tokenizer,
        max_points=max(1, int(scan_divergences)),
        min_index=max(0, int(min_divergence_index)),
        skip_style=skip_style_divergence,
        require_math_signal=require_math_signal_divergence,
        math_signal_window=math_signal_window,
    )
    if not divs:
        return None

    best_record: Optional[Dict[str, Any]] = None
    for scan_rank, div in enumerate(divs):
        prefix_at_div_ids_list = prefix_ids_list + student_draft[: div.index]
        prefix_at_div_ids = torch.tensor([prefix_at_div_ids_list], dtype=torch.long, device=device)
        stats = next_token_stats(student_model, prefix_at_div_ids, topk=student_topk)
        topk_ids = stats["topk_ids"]
        topk_logprobs = stats["topk_logprobs"]
        entropy = stats["entropy"]

        candidate_ids = _append_unique(topk_ids, [div.teacher_token_id])
        rollout_results: List[Dict[str, Any]] = []
        best_idx: Optional[int] = None
        best_score = -1.0
        candidate_seed_ids: List[List[int]] = []

        for idx, cid in enumerate(candidate_ids):
            seed_ids = [int(cid)]
            uses_teacher_span = int(cid) == int(div.teacher_token_id)
            if uses_teacher_span and teacher_span_len > 1:
                span_end = min(len(teacher_shadow), div.index + int(teacher_span_len))
                seed_ids.extend(int(x) for x in teacher_shadow[div.index + 1 : span_end])
            warm_ids: List[int] = []
            if uses_teacher_span and warm_start_len > 0:
                warm_prefix = torch.tensor([prefix_at_div_ids_list + seed_ids], dtype=torch.long, device=device)
                warm_ids = teacher_backend.generate_ids(tokenizer, warm_prefix, int(warm_start_len), temperature)
                seed_ids.extend(int(x) for x in warm_ids)
            candidate_seed_ids.append(seed_ids)

            rollout_prefix = torch.tensor([prefix_at_div_ids_list + seed_ids], dtype=torch.long, device=device)
            remaining_rollout = max(0, int(rollout_len) - len(seed_ids))
            rollout_new = greedy_generate_new_ids(
                student_model, rollout_prefix, remaining_rollout, eos_token_id=eos_id, temperature=temperature
            )
            generated_ids = seed_ids + rollout_new
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            full_text = tokenizer.decode(
                prefix_at_div_ids_list + generated_ids,
                skip_special_tokens=True,
            )
            verification = verify_text(generated_text, gold)
            score = verification.score
            rollout_results.append(
                {
                    "candidate_id": int(cid),
                    "candidate_text": tokenizer.decode([int(cid)], skip_special_tokens=False),
                    "seed_ids": seed_ids,
                    "seed_text": tokenizer.decode(seed_ids, skip_special_tokens=False),
                    "warm_start_ids": warm_ids,
                    "warm_start_text": tokenizer.decode(warm_ids, skip_special_tokens=False),
                    "generated_text": generated_text,
                    "rollout_text": full_text,
                    "parsed_answer": verification.parsed_answer,
                    "score": score,
                }
            )
            if score > best_score:
                best_score = score
                best_idx = idx

        if gold is None:
            label_idx = None
            decision = "unlabeled"
        elif best_idx is None or best_score < float(min_accept_score):
            label_idx = None
            decision = "abstain_no_verified_candidate"
        else:
            label_idx = int(best_idx)
            best_candidate = int(candidate_ids[label_idx])
            if best_candidate == int(div.teacher_token_id) and best_candidate != int(div.student_token_id):
                decision = "accept_teacher_span" if teacher_span_len > 1 or warm_start_len > 0 else "accept_teacher"
            elif best_candidate == int(div.student_token_id):
                decision = "keep_student"
            else:
                decision = "select_student_topk"

        candidate_features = build_candidate_features(
            candidate_ids=candidate_ids,
            student_topk_ids=topk_ids,
            student_topk_logprobs=topk_logprobs,
            student_token_id=div.student_token_id,
            teacher_token_id=div.teacher_token_id,
            entropy=entropy,
            divergence_index=div.index,
            draft_len=draft_len,
            tokenizer=tokenizer,
        )

        trigger_examples: List[Dict[str, Any]] = []
        trigger_examples.append(
            _trigger_example_for_position(
                student_model,
                tokenizer,
                device,
                prefix_ids_list,
                student_draft,
                div.index,
                draft_len,
                label=1,
                topk=student_topk,
            )
        )
        neg_positions = []
        for j in range(max(0, div.index - trigger_negative_window), min(len(student_draft), div.index + trigger_negative_window + 1)):
            if j != div.index:
                neg_positions.append(j)
        if 0 not in neg_positions and div.index != 0 and len(student_draft) > 0:
            neg_positions.append(0)
        for j in neg_positions:
            trigger_examples.append(
                _trigger_example_for_position(
                    student_model,
                    tokenizer,
                    device,
                    prefix_ids_list,
                    student_draft,
                    j,
                    draft_len,
                    label=0,
                    topk=student_topk,
                )
            )

        record = {
            "id": row.get("id"),
            "prompt": prompt,
            "gold": gold,
            "prefix_ids": prefix_ids_list,
            "prefix_text": tokenizer.decode(prefix_ids_list, skip_special_tokens=True),
            "prefix_at_div_ids": prefix_at_div_ids_list,
            "prefix_at_div_text": tokenizer.decode(prefix_at_div_ids_list, skip_special_tokens=True),
            "student_draft_ids": student_draft,
            "teacher_shadow_ids": teacher_shadow,
            "student_draft_text": tokenizer.decode(student_draft, skip_special_tokens=False),
            "teacher_shadow_text": tokenizer.decode(teacher_shadow, skip_special_tokens=False),
            "divergence_index": div.index,
            "divergence_scan_rank": scan_rank,
            "num_scanned_divergences": len(divs),
            "divergence_reason": div.reason,
            "student_token_id": int(div.student_token_id),
            "teacher_token_id": int(div.teacher_token_id),
            "student_token_text": div.student_text,
            "teacher_token_text": div.teacher_text,
            "student_topk_ids": topk_ids,
            "student_topk_logprobs": topk_logprobs,
            "student_entropy": entropy,
            "student_top1_margin": float(stats.get("top1_margin", 0.0)),
            "candidate_ids": candidate_ids,
            "candidate_texts": [tokenizer.decode([int(x)], skip_special_tokens=False) for x in candidate_ids],
            "candidate_seed_ids": candidate_seed_ids,
            "candidate_seed_texts": [tokenizer.decode(xs, skip_special_tokens=False) for xs in candidate_seed_ids],
            "candidate_features": candidate_features,
            "label_idx": label_idx,
            "label_candidate_id": int(candidate_ids[label_idx]) if label_idx is not None else None,
            "best_score": best_score,
            "accept_decision": decision,
            "teacher_span_len": int(teacher_span_len),
            "warm_start_len": int(warm_start_len),
            "require_math_signal_divergence": bool(require_math_signal_divergence),
            "math_signal_window": int(math_signal_window),
            "rollout_results": rollout_results,
            "trigger_examples": trigger_examples,
        }
        if label_idx is not None:
            return record
        if best_record is None or float(record["best_score"]) > float(best_record["best_score"]):
            best_record = record

    return best_record
