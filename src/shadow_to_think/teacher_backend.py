from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

import requests
import torch

from .generation import greedy_generate_new_ids


class TeacherBackend(Protocol):
    def generate_ids(self, tokenizer, prefix_ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.0) -> List[int]:
        ...


@dataclass
class TransformersTeacherBackend:
    """Teacher backend backed by an in-process HuggingFace causal LM."""

    model: object

    def generate_ids(self, tokenizer, prefix_ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.0) -> List[int]:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        return greedy_generate_new_ids(
            self.model,
            prefix_ids,
            max_new_tokens,
            eos_token_id=eos_id,
            temperature=temperature,
        )


@dataclass
class VLLMTeacherClient:
    """OpenAI-compatible vLLM teacher client.

    vLLM's recommended serving path exposes an OpenAI-compatible HTTP API via
    `vllm serve <model> ...`. This client uses `/v1/completions` so that the
    teacher performs ordinary continuation from the exact text prefix.
    """

    base_url: str = "http://localhost:8000/v1"
    model: str = "teacher"
    api_key: str = "EMPTY"
    timeout: float = 120.0

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def generate_text(self, prompt: str, max_new_tokens: int, temperature: float = 0.0) -> str:
        url = self.base_url.rstrip("/") + "/completions"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "n": 1,
        }
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"vLLM completion failed: HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        try:
            return data["choices"][0]["text"]
        except Exception as exc:
            raise RuntimeError(f"Unexpected vLLM completion payload: {data}") from exc

    def generate_ids(self, tokenizer, prefix_ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.0) -> List[int]:
        prompt = tokenizer.decode(prefix_ids.squeeze(0).tolist(), skip_special_tokens=False)
        text = self.generate_text(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        return tokenizer.encode(text, add_special_tokens=False)[:max_new_tokens]
