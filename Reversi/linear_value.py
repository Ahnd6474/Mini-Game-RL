"""A deliberately simple 64-input linear Reversi win predictor."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .env import BOARD_CELLS
from .symmetry import transform_action


class LinearWinPredictor(nn.Module):
    """Predict current-player win probability from 64 relative board values.

    Bias is disabled so swapping the players negates the logit exactly:
    ``f(-board) == -f(board)``.  This property is required by negamax search.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(BOARD_CELLS, 1, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, relative_board: Tensor) -> Tensor:
        return self.linear(relative_board.to(self.linear.weight.dtype)).squeeze(-1)

    def probability(self, relative_board: Tensor) -> Tensor:
        return self(relative_board).sigmoid()

    @torch.no_grad()
    def symmetrize_(self) -> None:
        """Tie all D4-equivalent cells by averaging their learned weights."""
        weights = self.linear.weight[0]
        symmetric = torch.empty_like(weights)
        visited: set[int] = set()
        for action in range(BOARD_CELLS):
            if action in visited:
                continue
            orbit = sorted({
                transform_action(action, symmetry) for symmetry in range(8)
            })
            symmetric[orbit] = weights[orbit].mean()
            visited.update(orbit)
        weights.copy_(symmetric)

    def weights(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.linear.weight[0].detach().cpu())


def linear_value_loss(logits: Tensor, outcomes: Tensor) -> Tensor:
    """BCE loss where outcomes are loss=-1, draw=0, win=1."""
    targets = (outcomes.to(logits.dtype) + 1.0) * 0.5
    return F.binary_cross_entropy_with_logits(logits, targets)


def save_linear_evaluator(
    model: LinearWinPredictor,
    path: str | Path,
    metrics: dict[str, object] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "reversi-linear-value-v1",
        "weights": model.weights(),
        "metrics": metrics or {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_linear_weights(path: str | Path) -> tuple[float, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = tuple(float(value) for value in payload["weights"])
    if len(weights) != BOARD_CELLS or not all(math.isfinite(x) for x in weights):
        raise ValueError("linear evaluator must contain 64 finite weights")
    return weights


def relative_board(board: Sequence[int], player: int) -> tuple[int, ...]:
    if len(board) != BOARD_CELLS:
        raise ValueError("board must contain 64 cells")
    if player not in (-1, 1):
        raise ValueError("player must be -1 or 1")
    return tuple(int(cell) * player for cell in board)
