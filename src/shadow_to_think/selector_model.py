from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureSelector(nn.Module):
    """A tiny candidate selector for the MVP.

    It scores each candidate token independently using cheap scalar features.
    For every training record we apply softmax over that record's candidate set.
    """

    def __init__(self, feature_dim: int = 8, hidden_size: int = 128, dropout: float = 0.05):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_size = int(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        """Return candidate scores.

        Args:
            candidate_features: [num_candidates, feature_dim]
        Returns:
            scores: [num_candidates]
        """
        if candidate_features.ndim != 2:
            raise ValueError("candidate_features must be [num_candidates, feature_dim]")
        return self.net(candidate_features).squeeze(-1)


def selector_loss(model: FeatureSelector, features: torch.Tensor, label_idx: int) -> torch.Tensor:
    scores = model(features).unsqueeze(0)  # [1, C]
    target = torch.tensor([label_idx], dtype=torch.long, device=features.device)
    return F.cross_entropy(scores, target)


@torch.no_grad()
def selector_predict(model: FeatureSelector, features: torch.Tensor) -> int:
    scores = model(features)
    return int(torch.argmax(scores).item())


def save_selector(model: FeatureSelector, path: str, extra: Dict | None = None) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "feature_dim": model.feature_dim,
        "hidden_size": model.hidden_size,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_selector(path: str, map_location="cpu") -> FeatureSelector:
    payload = torch.load(path, map_location=map_location)
    model = FeatureSelector(feature_dim=payload["feature_dim"], hidden_size=payload["hidden_size"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
