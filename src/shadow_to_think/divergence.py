from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence


PUNCT_RE = re.compile(r"^[\s\.,;:!\?\-–—\)\]\}\(\[\{\"'`]+$")

DEFAULT_STOPLIKE = {
    "", " ", "\n", "\t", ".", ",", ";", ":", "!", "?", "-", "—", "–",
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or",
}

STYLELIKE = {
    "to", "let", "we", "first", "second", "next", "then", "now", "the",
    "solve", "find", "determine", "calculate", "compute", "figure",
    "problem", "question", "answer", "step", "given",
}

MATH_SIGNAL_RE = re.compile(r"[\d=+\-*/^<>\\$%]")

NEAR_SYNONYM_GROUPS = [
    {"therefore", "thus", "hence", "so"},
    {"because", "since"},
    {"however", "but", "nevertheless", "though"},
    {"answer", "result", "solution"},
]


@dataclass
class DivergencePoint:
    index: int
    student_token_id: int
    teacher_token_id: int
    student_text: str
    teacher_text: str
    reason: str


def normalize_token_text(text: str) -> str:
    return text.strip().lower().replace("Ġ", "").replace("▁", "")


def is_pure_punct_or_space(text: str) -> bool:
    return bool(PUNCT_RE.match(text))


def same_near_synonym(a: str, b: str) -> bool:
    na, nb = normalize_token_text(a), normalize_token_text(b)
    if na == nb:
        return True
    for group in NEAR_SYNONYM_GROUPS:
        if na in group and nb in group:
            return True
    return False


def token_is_meaningful(student_text: str, teacher_text: str) -> bool:
    """Heuristic filter for first meaningful divergence.

    V1 deliberately keeps this simple. Later versions can replace this with a
    learned divergence classifier or span-level semantic alignment.
    """
    s = normalize_token_text(student_text)
    t = normalize_token_text(teacher_text)
    if s == t:
        return False
    if is_pure_punct_or_space(student_text) or is_pure_punct_or_space(teacher_text):
        return False
    if s in DEFAULT_STOPLIKE and t in DEFAULT_STOPLIKE:
        return False
    if same_near_synonym(student_text, teacher_text):
        return False
    return True


def token_has_math_signal(text: str) -> bool:
    stripped = normalize_token_text(text)
    if bool(MATH_SIGNAL_RE.search(text)):
        return True
    return stripped in {
        "sum", "difference", "product", "ratio", "percent", "remainder",
        "divided", "times", "plus", "minus", "equals", "equal", "half",
        "twice", "total", "average", "probability", "integer",
    }


def span_has_math_signal(ids: Sequence[int], tokenizer, start: int, window: int) -> bool:
    end = min(len(ids), max(0, int(start)) + max(1, int(window)))
    for tid in ids[max(0, int(start)) : end]:
        if token_has_math_signal(tokenizer.decode([int(tid)], skip_special_tokens=False)):
            return True
    return False


def is_style_divergence(student_text: str, teacher_text: str) -> bool:
    """Return True for wording/style divergences that rarely affect math state."""
    if token_has_math_signal(student_text) or token_has_math_signal(teacher_text):
        return False
    s = normalize_token_text(student_text)
    t = normalize_token_text(teacher_text)
    return s in STYLELIKE or t in STYLELIKE


def find_meaningful_divergences(
    student_ids: Sequence[int],
    teacher_ids: Sequence[int],
    tokenizer,
    *,
    max_points: int | None = None,
    min_index: int = 0,
    skip_style: bool = False,
    require_math_signal: bool = False,
    math_signal_window: int = 8,
) -> List[DivergencePoint]:
    points: List[DivergencePoint] = []
    limit = min(len(student_ids), len(teacher_ids))
    for i in range(max(0, int(min_index)), limit):
        sid, tid = int(student_ids[i]), int(teacher_ids[i])
        if sid == tid:
            continue
        s_text = tokenizer.decode([sid], skip_special_tokens=False)
        t_text = tokenizer.decode([tid], skip_special_tokens=False)
        if not token_is_meaningful(s_text, t_text):
            continue
        if skip_style and is_style_divergence(s_text, t_text):
            continue
        if require_math_signal and not (
            span_has_math_signal(student_ids, tokenizer, i, math_signal_window)
            or span_has_math_signal(teacher_ids, tokenizer, i, math_signal_window)
        ):
            continue
        points.append(
            DivergencePoint(
                index=i,
                student_token_id=sid,
                teacher_token_id=tid,
                student_text=s_text,
                teacher_text=t_text,
                reason="first_nontrivial_token_mismatch" if not points else "later_nontrivial_token_mismatch",
            )
        )
        if max_points is not None and len(points) >= int(max_points):
            break
    return points


def find_first_meaningful_divergence(
    student_ids: Sequence[int],
    teacher_ids: Sequence[int],
    tokenizer,
) -> Optional[DivergencePoint]:
    points = find_meaningful_divergences(student_ids, teacher_ids, tokenizer, max_points=1)
    return points[0] if points else None
