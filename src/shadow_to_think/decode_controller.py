from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from .divergence import find_first_meaningful_divergence, find_meaningful_divergences
from .features import build_candidate_features, build_trigger_features
from .generation import encode_prompt, greedy_generate_new_ids, last_hidden_state, next_token_stats
from .hidden_selector_model import HiddenStateSelector, hidden_selector_predict
from .teacher_backend import TeacherBackend
from .trigger_model import FeatureTrigger, trigger_score


def _append_unique(base: List[int], extra: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in list(base) + list(extra):
        x = int(x)
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


STEP_BOUNDARY_RE = re.compile(r"(\n|(?:^|[\s])(?:therefore|thus|hence|so),?\s|[.;:]\s*$)", re.IGNORECASE)


@dataclass
class DecodeConfig:
    max_new_tokens: int = 256
    draft_len: int = 32
    shadow_len: int = 32
    student_topk: int = 16
    temperature: float = 0.0
    trigger_threshold: float = 0.5
    mode: str = "shadow"  # shadow | local
    max_prompt_tokens: int = 2048
    teacher_span_len: int = 1
    skip_style_divergence: bool = False
    min_divergence_index: int = 0
    intervention_policy: str = "selector"  # selector | teacher_only
    teacher_span_mode: str = "fixed"  # fixed | step
    teacher_span_min_len: int = 4
    require_math_signal_divergence: bool = False
    math_signal_window: int = 8
    max_teacher_calls: int | None = None
    trace_decisions: bool = False


class ShadowDecodeController:
    """Generation-time Shadow-to-Think controller.

    Modes:
      - shadow: call teacher backend on each chunk, locate first meaningful
        divergence, then use hidden selector or teacher token at the divergence.
      - local: student-only decoding; local trigger decides risky positions and
        hidden selector reranks student top-K without teacher.
    """

    def __init__(
        self,
        student_model,
        tokenizer,
        *,
        device,
        teacher_backend: TeacherBackend | None = None,
        trigger: FeatureTrigger | None = None,
        hidden_selector: HiddenStateSelector | None = None,
        config: DecodeConfig | None = None,
    ):
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.device = device
        self.teacher_backend = teacher_backend
        self.trigger = trigger
        self.hidden_selector = hidden_selector
        self.config = config or DecodeConfig()
        self.student_model.eval()
        if self.trigger is not None:
            self.trigger.eval()
        if self.hidden_selector is not None:
            self.hidden_selector.eval()

    @torch.no_grad()
    def _select_candidate(self, prefix_ids: torch.Tensor, candidate_ids: List[int], candidate_features: List[List[float]]) -> int:
        if self.hidden_selector is None:
            return int(candidate_ids[0])
        h = last_hidden_state(self.student_model, prefix_ids).to(self.device)
        cand_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=self.device)
        embeds = self.student_model.get_input_embeddings()(cand_tensor).detach().float()
        feats = torch.tensor(candidate_features, dtype=torch.float32, device=self.device)
        idx = hidden_selector_predict(self.hidden_selector.to(self.device), h, embeds, feats)
        return int(candidate_ids[idx])

    @torch.no_grad()
    def _trigger_decision(self, prefix_ids: torch.Tensor, position_index: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
        if self.trigger is None:
            return True, None
        stats = next_token_stats(self.student_model, prefix_ids, topk=self.config.student_topk)
        feats = build_trigger_features(
            stats=stats,
            position_index=position_index,
            draft_len=max(self.config.draft_len, 1),
            next_token_id=stats["top1_id"],
            tokenizer=self.tokenizer,
        )
        score = trigger_score(self.trigger.to(self.device), torch.tensor(feats, dtype=torch.float32, device=self.device))
        risky = score >= self.config.trigger_threshold
        return risky, {
            "position_index": int(position_index),
            "trigger_score": float(score),
            "trigger_threshold": float(self.config.trigger_threshold),
            "trigger_features": [float(x) for x in feats],
            "entropy": float(stats["entropy"]),
            "top1_margin": float(stats["top1_margin"]),
            "top1_id": int(stats["top1_id"]),
            "top1_text": self.tokenizer.decode([int(stats["top1_id"])], skip_special_tokens=False),
            "risky": bool(risky),
        }

    @torch.no_grad()
    def _trigger_risky(self, prefix_ids: torch.Tensor, position_index: int) -> bool:
        risky, _ = self._trigger_decision(prefix_ids, position_index)
        return risky

    def _teacher_span_ids(self, teacher_shadow: List[int], div_index: int) -> List[int]:
        """Return the teacher span to inject, including the divergence token."""
        max_len = max(1, int(self.config.teacher_span_len))
        fixed_end = min(len(teacher_shadow), int(div_index) + max_len)
        fixed_span = [int(x) for x in teacher_shadow[int(div_index) : fixed_end]]
        if self.config.teacher_span_mode != "step" or max_len <= 1:
            return fixed_span

        min_len = max(1, min(int(self.config.teacher_span_min_len), max_len))
        best = fixed_span
        for span_len in range(min_len, len(fixed_span) + 1):
            candidate = fixed_span[:span_len]
            text = self.tokenizer.decode(candidate, skip_special_tokens=False)
            if STEP_BOUNDARY_RE.search(text):
                return candidate
        return best

    @torch.no_grad()
    def generate(self, prompt: str) -> Dict:
        if self.config.mode == "local":
            return self._generate_local(prompt)
        if self.teacher_backend is None:
            raise ValueError("shadow mode requires teacher_backend")
        return self._generate_shadow(prompt)

    @torch.no_grad()
    def _generate_local(self, prompt: str) -> Dict:
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        cur = encode_prompt(self.tokenizer, prompt, max_prompt_tokens=self.config.max_prompt_tokens, device=self.device)
        generated: List[int] = []
        interventions = []
        for pos in range(self.config.max_new_tokens):
            stats = next_token_stats(self.student_model, cur, topk=self.config.student_topk)
            token_id = int(stats["top1_id"])
            if self._trigger_risky(cur, pos) and self.hidden_selector is not None:
                candidate_ids = stats["topk_ids"]
                feats = build_candidate_features(
                    candidate_ids=candidate_ids,
                    student_topk_ids=stats["topk_ids"],
                    student_topk_logprobs=stats["topk_logprobs"],
                    student_token_id=stats["top1_id"],
                    teacher_token_id=-999999,  # no teacher in local mode
                    entropy=stats["entropy"],
                    divergence_index=pos % max(self.config.draft_len, 1),
                    draft_len=max(self.config.draft_len, 1),
                    tokenizer=self.tokenizer,
                )
                selected = self._select_candidate(cur, candidate_ids, feats)
                if selected != token_id:
                    interventions.append({"position": pos, "from": token_id, "to": selected, "mode": "local_selector"})
                token_id = selected
            generated.append(token_id)
            next_tensor = torch.tensor([[token_id]], dtype=torch.long, device=cur.device)
            cur = torch.cat([cur, next_tensor], dim=1)
            if eos_id is not None and token_id == eos_id:
                break
        return {
            "text": self.tokenizer.decode(generated, skip_special_tokens=True),
            "generated_ids": generated,
            "interventions": interventions,
            "teacher_calls": 0,
        }

    @torch.no_grad()
    def _generate_shadow(self, prompt: str) -> Dict:
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        cur = encode_prompt(self.tokenizer, prompt, max_prompt_tokens=self.config.max_prompt_tokens, device=self.device)
        generated: List[int] = []
        interventions = []
        decision_trace = []
        teacher_calls = 0
        while len(generated) < self.config.max_new_tokens:
            if self.config.max_teacher_calls is not None and teacher_calls >= int(self.config.max_teacher_calls):
                remaining = self.config.max_new_tokens - len(generated)
                rest = greedy_generate_new_ids(
                    self.student_model,
                    cur,
                    remaining,
                    eos_token_id=eos_id,
                    temperature=self.config.temperature,
                )
                generated.extend(rest)
                if rest:
                    cur = torch.tensor([cur.squeeze(0).tolist() + rest], dtype=torch.long, device=self.device)
                break
            # The trigger is trained on positions inside a draft, so evaluate it
            # before each accepted student token instead of skipping a whole chunk.
            trigger_position = len(generated) % max(self.config.draft_len, 1)
            risky, trace = self._trigger_decision(cur, trigger_position)
            if trace is not None and self.config.trace_decisions:
                trace.update({"absolute_position": len(generated), "action": "call_teacher" if risky else "keep_student"})
                decision_trace.append(trace)
            if self.trigger is not None and not risky:
                token = greedy_generate_new_ids(
                    self.student_model,
                    cur,
                    1,
                    eos_token_id=eos_id,
                    temperature=self.config.temperature,
                )
                if not token:
                    break
                token_id = int(token[0])
                generated.append(token_id)
                cur = torch.cat([cur, torch.tensor([[token_id]], dtype=torch.long, device=self.device)], dim=1)
                if eos_id is not None and token_id == eos_id:
                    break
                continue

            chunk_len = min(self.config.draft_len, self.config.max_new_tokens - len(generated))
            student_draft = greedy_generate_new_ids(
                self.student_model,
                cur,
                chunk_len,
                eos_token_id=eos_id,
                temperature=self.config.temperature,
            )
            if not student_draft:
                break
            teacher_shadow = self.teacher_backend.generate_ids(self.tokenizer, cur, self.config.shadow_len, self.config.temperature)
            teacher_calls += 1
            if (
                self.config.skip_style_divergence
                or self.config.min_divergence_index > 0
                or self.config.require_math_signal_divergence
            ):
                divs = find_meaningful_divergences(
                    student_draft,
                    teacher_shadow,
                    self.tokenizer,
                    max_points=1,
                    min_index=self.config.min_divergence_index,
                    skip_style=self.config.skip_style_divergence,
                    require_math_signal=self.config.require_math_signal_divergence,
                    math_signal_window=self.config.math_signal_window,
                )
                div = divs[0] if divs else None
            else:
                div = find_first_meaningful_divergence(student_draft, teacher_shadow, self.tokenizer)
            if div is None:
                generated.extend(student_draft)
                cur = torch.tensor([cur.squeeze(0).tolist() + student_draft], dtype=torch.long, device=self.device)
                if eos_id is not None and eos_id in student_draft:
                    break
                continue

            before = student_draft[: div.index]
            prefix_at_div_list = cur.squeeze(0).tolist() + before
            prefix_at_div = torch.tensor([prefix_at_div_list], dtype=torch.long, device=self.device)
            stats = next_token_stats(self.student_model, prefix_at_div, topk=self.config.student_topk)
            candidate_ids = _append_unique(stats["topk_ids"], [div.teacher_token_id])
            feats = build_candidate_features(
                candidate_ids=candidate_ids,
                student_topk_ids=stats["topk_ids"],
                student_topk_logprobs=stats["topk_logprobs"],
                student_token_id=div.student_token_id,
                teacher_token_id=div.teacher_token_id,
                entropy=stats["entropy"],
                divergence_index=div.index,
                draft_len=self.config.draft_len,
                tokenizer=self.tokenizer,
            )
            selected = self._select_candidate(prefix_at_div, candidate_ids, feats) if self.hidden_selector is not None else int(div.teacher_token_id)
            if self.config.intervention_policy == "teacher_only" and int(selected) != int(div.teacher_token_id):
                selected = int(div.student_token_id)
            if int(selected) == int(div.teacher_token_id):
                selected_ids = self._teacher_span_ids(teacher_shadow, div.index)
            else:
                selected_ids = [int(selected)]
            remaining_slots = self.config.max_new_tokens - len(generated) - len(before)
            selected_ids = selected_ids[: max(1, remaining_slots)]
            generated.extend(before + selected_ids)
            interventions.append(
                {
                    "position": len(generated) - len(selected_ids),
                    "divergence_index": div.index,
                    "student_token_id": int(div.student_token_id),
                    "teacher_token_id": int(div.teacher_token_id),
                    "selected_token_id": int(selected),
                    "selected_span_ids": selected_ids,
                    "student_token_text": div.student_text,
                    "teacher_token_text": div.teacher_text,
                    "selected_token_text": self.tokenizer.decode([int(selected)], skip_special_tokens=False),
                    "selected_span_text": self.tokenizer.decode(selected_ids, skip_special_tokens=False),
                    "teacher_span_mode": self.config.teacher_span_mode,
                }
            )
            cur = torch.tensor([prefix_at_div_list + selected_ids], dtype=torch.long, device=self.device)
            if eos_id is not None and eos_id in selected_ids:
                break
        return {
            "text": self.tokenizer.decode(generated, skip_special_tokens=True),
            "generated_ids": generated,
            "interventions": interventions,
            "teacher_calls": teacher_calls,
            "decision_trace": decision_trace,
        }
