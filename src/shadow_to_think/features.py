from __future__ import annotations

from typing import Dict, List, Sequence


def build_candidate_features(
    candidate_ids: Sequence[int],
    student_topk_ids: Sequence[int],
    student_topk_logprobs: Sequence[float],
    student_token_id: int,
    teacher_token_id: int,
    entropy: float,
    divergence_index: int,
    draft_len: int,
    tokenizer=None,
) -> List[List[float]]:
    """Build lightweight scalar features for candidate ranking."""
    logprob_by_id = {int(tid): float(lp) for tid, lp in zip(student_topk_ids, student_topk_logprobs)}
    rank_by_id = {int(tid): i for i, tid in enumerate(student_topk_ids)}
    feats: List[List[float]] = []
    denom = max(len(student_topk_ids), 1)
    rel_pos = float(divergence_index) / float(max(draft_len, 1))
    top1_id = int(student_topk_ids[0]) if student_topk_ids else -1
    for cid in candidate_ids:
        cid = int(cid)
        rank = rank_by_id.get(cid, len(student_topk_ids))
        logprob = logprob_by_id.get(cid, -30.0)
        token_text = tokenizer.decode([cid], skip_special_tokens=False) if tokenizer is not None else ""
        token_len = min(len(token_text), 16) / 16.0
        feats.append([
            logprob / 30.0,                       # roughly [-1, 0]
            float(rank) / float(denom),           # [0, 1+] rank
            1.0 if cid == top1_id else 0.0,
            1.0 if cid == int(student_token_id) else 0.0,
            1.0 if cid == int(teacher_token_id) else 0.0,
            min(float(entropy) / 20.0, 1.0),
            rel_pos,
            token_len,
        ])
    return feats


FEATURE_NAMES = [
    "student_logprob_scaled",
    "student_rank_scaled",
    "is_student_top1",
    "is_original_student_token",
    "is_teacher_token",
    "student_entropy_scaled",
    "divergence_relative_position",
    "token_text_length_scaled",
]


def build_trigger_features(
    *,
    stats: Dict,
    position_index: int,
    draft_len: int,
    next_token_id: int | None = None,
    tokenizer=None,
) -> List[float]:
    """Build local trigger features using only student-side information.

    The feature vector intentionally avoids teacher signals so that it can be
    used during student-only decoding.
    """
    topk_ids = [int(x) for x in stats.get("topk_ids", [])]
    topk_logprobs = [float(x) for x in stats.get("topk_logprobs", [])]
    top1_lp = float(stats.get("top1_logprob", topk_logprobs[0] if topk_logprobs else -30.0))
    top2_lp = float(topk_logprobs[1]) if len(topk_logprobs) > 1 else top1_lp
    margin = float(stats.get("top1_margin", top1_lp - top2_lp))
    entropy = float(stats.get("entropy", 0.0))
    top1_prob = float(stats.get("top1_prob", 0.0))
    rel_pos = float(position_index) / float(max(draft_len, 1))
    token_id = int(next_token_id if next_token_id is not None else (topk_ids[0] if topk_ids else -1))
    token_text = tokenizer.decode([token_id], skip_special_tokens=False) if tokenizer is not None and token_id >= 0 else ""
    stripped = token_text.strip().lower()
    token_len = min(len(token_text), 16) / 16.0
    is_digit = 1.0 if any(ch.isdigit() for ch in token_text) else 0.0
    is_constraint = 1.0 if stripped in {"must", "cannot", "always", "only", "never", "yes", "no"} else 0.0
    is_reasoning = 1.0 if stripped in {"therefore", "however", "because", "thus", "hence", "but", "so"} else 0.0
    return [
        min(entropy / 20.0, 1.0),
        max(min(margin / 10.0, 1.0), -1.0),
        max(min(top1_lp / 30.0, 0.0), -1.0),
        max(min(top1_prob, 1.0), 0.0),
        rel_pos,
        token_len,
        is_digit,
        is_constraint,
        is_reasoning,
    ]


TRIGGER_FEATURE_NAMES = [
    "entropy_scaled",
    "top1_margin_scaled",
    "top1_logprob_scaled",
    "top1_prob",
    "relative_position",
    "token_text_length_scaled",
    "token_has_digit",
    "token_is_constraint",
    "token_is_reasoning_connector",
]
