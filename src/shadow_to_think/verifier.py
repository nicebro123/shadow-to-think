from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:/[1-9]\d*)?")
REPEATED_PROMPT_RE = re.compile(r"(?<=\S)\s+(?:Question|Problem)\s*:", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}")
ANSWER_MARKER_RE = re.compile(r"(?:final\s+answer|answer)\s*:|answer\s+is|答案是", re.IGNORECASE)


@dataclass
class VerificationResult:
    score: float
    parsed_answer: Optional[str]
    mode: str


def normalize_answer(text: str) -> str:
    text = str(text).strip()
    boxed = extract_boxed(text)
    if boxed is not None:
        text = boxed
    text = text.replace("$", "")
    text = re.sub(r"\\(?:left|right)", "", text)
    text = re.sub(r"[\s`。．,，;；:：.]+$", "", text)
    if re.fullmatch(r"[-+]?\d+\.0+", text):
        text = text.split(".", 1)[0]
    return re.sub(r"\s+", "", text.strip().lower())


def extract_boxed(text: str) -> Optional[str]:
    matches = BOXED_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def extract_last_number(text: str) -> Optional[str]:
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    return matches[-1]


def extract_answer(text: str) -> str:
    """Simple answer extractor for MVP math/code-style JSONL.

    It handles common forms like '#### 42' and otherwise falls back to the last
    number if present; if no number appears, it uses the last non-empty line.
    """
    repeated_prompt = REPEATED_PROMPT_RE.search(text)
    if repeated_prompt:
        text = text[: repeated_prompt.start()].strip()
    if "####" in text:
        tail = text.split("####")[-1].strip()
        if tail:
            text = tail
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    markers = list(ANSWER_MARKER_RE.finditer(text))
    if markers:
        tail = text[markers[-1].end() :].strip()
        if tail:
            text = tail
    number = extract_last_number(text)
    if number is not None:
        return number
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def verify_text(text: str, gold: Optional[str], mode: str = "exact_or_last_number") -> VerificationResult:
    if gold is None:
        return VerificationResult(score=0.0, parsed_answer=None, mode="no_gold")
    pred = extract_answer(text)
    score = 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0
    return VerificationResult(score=score, parsed_answer=pred, mode=mode)
