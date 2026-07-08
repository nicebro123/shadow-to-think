from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F


def get_model_device(model) -> torch.device:
    """Best-effort device lookup that also works for normal HF models."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def greedy_generate_new_ids(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 0.0,
) -> List[int]:
    """Generate new token ids from a model. Greedy by default."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, seq_len]")
    generated: List[int] = []
    cur = input_ids.to(get_model_device(model))
    for _ in range(max_new_tokens):
        out = model(cur)
        logits = out.logits[:, -1, :]
        if temperature and temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        token_id = int(next_id.item())
        generated.append(token_id)
        cur = torch.cat([cur, next_id.to(cur.device)], dim=1)
        if eos_token_id is not None and token_id == eos_token_id:
            break
    return generated


@torch.no_grad()
def next_token_stats(model, prefix_ids: torch.Tensor, topk: int = 16) -> Dict:
    """Return next-token top-k, logprobs, probabilities, entropy and margin for a prefix."""
    prefix_ids = prefix_ids.to(get_model_device(model))
    out = model(prefix_ids)
    logits = out.logits[:, -1, :]
    logprobs = F.log_softmax(logits, dim=-1).squeeze(0)
    probs = logprobs.exp()
    entropy = float(-(probs * logprobs).sum().item())
    topk = min(int(topk), logprobs.numel())
    vals, ids = torch.topk(logprobs, k=topk)
    top_probs = vals.exp()
    if topk >= 2:
        margin = float((vals[0] - vals[1]).item())
    else:
        margin = 0.0
    return {
        "topk_ids": [int(x) for x in ids.tolist()],
        "topk_logprobs": [float(x) for x in vals.tolist()],
        "topk_probs": [float(x) for x in top_probs.tolist()],
        "entropy": entropy,
        "top1_margin": margin,
        "top1_id": int(ids[0].item()),
        "top1_logprob": float(vals[0].item()),
        "top1_prob": float(top_probs[0].item()),
        "logprobs_tensor": logprobs.detach().cpu(),
    }


@torch.no_grad()
def last_hidden_state(model, prefix_ids: torch.Tensor) -> torch.Tensor:
    """Return detached last-layer hidden state for the final prefix token: [hidden_dim]."""
    prefix_ids = prefix_ids.to(get_model_device(model))
    out = model(prefix_ids, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states[-1][:, -1, :].squeeze(0)
    return hidden.detach().float().cpu()


def encode_prompt(tokenizer, prompt: str, max_prompt_tokens: int | None = None, device=None) -> torch.Tensor:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if max_prompt_tokens is not None and len(ids) > max_prompt_tokens:
        ids = ids[-max_prompt_tokens:]
    if not ids:
        ids = [tokenizer.eos_token_id]
    tensor = torch.tensor([ids], dtype=torch.long)
    if device is not None:
        tensor = tensor.to(device)
    return tensor
