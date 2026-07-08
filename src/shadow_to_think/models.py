from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class LoadedLM:
    model: object
    tokenizer: object
    device: torch.device


def load_lm(
    model_name_or_path: str,
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = True,
) -> LoadedLM:
    """Load a causal LM and tokenizer with conservative defaults.

    This function is intentionally simple for the MVP. It works with normal
    HuggingFace causal LMs and can be replaced later by vLLM serving for the
    teacher side.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = None
    if dtype == "auto":
        torch_dtype = "auto"
    elif dtype in {"bf16", "bfloat16"}:
        torch_dtype = torch.bfloat16
    elif dtype in {"fp16", "float16"}:
        torch_dtype = torch.float16
    elif dtype in {"fp32", "float32"}:
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    if device == "auto":
        device_map = "auto"
        actual_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_map = None
        actual_device = torch.device(device)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    if device_map is None:
        model = model.to(actual_device)
    model.eval()
    return LoadedLM(model=model, tokenizer=tokenizer, device=actual_device)


def ensure_same_tokenizer(student_tokenizer: object, teacher_tokenizer: object) -> None:
    """Best-effort tokenizer compatibility check.

    V1 assumes same tokenizer. We do not require identical Python classes, but
    vocab size and a few common encodings should match.
    """
    if getattr(student_tokenizer, "vocab_size", None) != getattr(teacher_tokenizer, "vocab_size", None):
        raise ValueError(
            "V1 expects same tokenizer/vocab size. Use same-family models or implement span-level alignment."
        )
    probes = [" therefore", " however", " answer", " 42", "\n"]
    for text in probes:
        if student_tokenizer.encode(text, add_special_tokens=False) != teacher_tokenizer.encode(text, add_special_tokens=False):
            raise ValueError(
                "Student/teacher tokenizers appear incompatible. V1 requires same tokenizer."
            )
