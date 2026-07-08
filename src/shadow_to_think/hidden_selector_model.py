from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class HiddenStateSelector(nn.Module):
    """Candidate selector based on student hidden state and candidate embeddings."""

    def __init__(
        self,
        hidden_dim: int,
        feature_dim: int = 8,
        selector_dim: int = 256,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.feature_dim = int(feature_dim)
        self.selector_dim = int(selector_dim)
        in_dim = self.hidden_dim * 3 + self.feature_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.selector_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.selector_dim, self.selector_dim),
            nn.GELU(),
            nn.Linear(self.selector_dim, 1),
        )

    def forward(
        self,
        prefix_hidden: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score candidates.

        Args:
            prefix_hidden: [H]
            candidate_embeddings: [C, H]
            candidate_features: [C, F], optional
        Returns:
            scores: [C]
        """
        if prefix_hidden.ndim != 1:
            raise ValueError("prefix_hidden must be [hidden_dim]")
        if candidate_embeddings.ndim != 2:
            raise ValueError("candidate_embeddings must be [num_candidates, hidden_dim]")
        c = candidate_embeddings.shape[0]
        h = prefix_hidden.unsqueeze(0).expand(c, -1)
        if candidate_features is None:
            candidate_features = torch.zeros(c, self.feature_dim, device=candidate_embeddings.device)
        if candidate_features.ndim != 2:
            raise ValueError("candidate_features must be [num_candidates, feature_dim]")
        x = torch.cat([h, candidate_embeddings, h * candidate_embeddings, candidate_features], dim=-1)
        return self.net(x).squeeze(-1)


def hidden_selector_loss(
    model: HiddenStateSelector,
    prefix_hidden: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    candidate_features: torch.Tensor,
    label_idx: int,
) -> torch.Tensor:
    scores = model(prefix_hidden, candidate_embeddings, candidate_features).unsqueeze(0)
    target = torch.tensor([int(label_idx)], dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores, target)


@torch.no_grad()
def hidden_selector_predict(
    model: HiddenStateSelector,
    prefix_hidden: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    candidate_features: torch.Tensor,
) -> int:
    scores = model(prefix_hidden, candidate_embeddings, candidate_features)
    return int(torch.argmax(scores).item())


def save_hidden_selector(model: HiddenStateSelector, path: str, extra: Dict | None = None) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "hidden_dim": model.hidden_dim,
        "feature_dim": model.feature_dim,
        "selector_dim": model.selector_dim,
        "extra": extra or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_hidden_selector(path: str, map_location="cpu") -> HiddenStateSelector:
    payload = torch.load(path, map_location=map_location)
    model = HiddenStateSelector(
        hidden_dim=payload["hidden_dim"],
        feature_dim=payload["feature_dim"],
        selector_dim=payload["selector_dim"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
