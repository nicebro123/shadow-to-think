from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTrigger(nn.Module):
    """Student-side local trigger: predict whether a token position is risky."""

    def __init__(self, feature_dim: int = 9, hidden_size: int = 64, dropout: float = 0.05):
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        return self.net(features).squeeze(-1)


@torch.no_grad()
def trigger_score(model: FeatureTrigger, features: torch.Tensor) -> float:
    model.eval()
    logits = model(features)
    return float(torch.sigmoid(logits).view(-1)[0].item())


def trigger_loss(model: FeatureTrigger, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = model(features)
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def save_trigger(model: FeatureTrigger, path: str, extra: Dict | None = None) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "feature_dim": model.feature_dim,
        "hidden_size": model.hidden_size,
        "extra": extra or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_trigger(path: str, map_location="cpu") -> FeatureTrigger:
    payload = torch.load(path, map_location=map_location)
    model = FeatureTrigger(feature_dim=payload["feature_dim"], hidden_size=payload["hidden_size"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
