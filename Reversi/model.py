"""Independent DVN (critic) and DPN (policy) Transformer models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .env import BOARD_CELLS, BOARD_SIZE

FINAL_OWN = 0
FINAL_OPPONENT = 1
FINAL_EMPTY = 2


@dataclass(frozen=True, slots=True)
class ReversiTransformerConfig:
    d_model: int = 64
    num_layers: int = 3
    num_heads: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")


@dataclass(slots=True)
class DVNOutput:
    final_board_logits: Tensor
    value: Tensor


@dataclass(slots=True)
class DPNOutput:
    policy_logits: Tensor


class BoardEncoder(nn.Module):
    """Encoder class instantiated independently inside DVN and DPN."""

    def __init__(self, config: ReversiTransformerConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model
        self.cell_projection = nn.Linear(4, d_model)
        self.row_embedding = nn.Embedding(BOARD_SIZE, d_model)
        self.col_embedding = nn.Embedding(BOARD_SIZE, d_model)
        self.empty_count_embedding = nn.Embedding(BOARD_CELLS + 1, d_model)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            # tanh GELU can be reproduced efficiently by standalone NumPy
            # inference in the generated submission source.
            activation=nn.GELU(approximate="tanh"),
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        row_ids = torch.arange(BOARD_SIZE).repeat_interleave(BOARD_SIZE)
        col_ids = torch.arange(BOARD_SIZE).repeat(BOARD_SIZE)
        self.register_buffer("row_ids", row_ids, persistent=False)
        self.register_buffer("col_ids", col_ids, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.row_embedding.weight, std=0.02)
        nn.init.normal_(self.col_embedding.weight, std=0.02)
        nn.init.normal_(self.empty_count_embedding.weight, std=0.02)

    def forward(
        self,
        board: Tensor,
        current_player: Tensor,
        legal_action_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        _validate_position_inputs(board, current_player, legal_action_mask)
        player = current_player[:, None]
        empty = board.eq(0)
        features = (
            empty,
            board.eq(player),
            board.eq(-player),
            torch.zeros_like(empty)
            if legal_action_mask is None
            else legal_action_mask[:, :BOARD_CELLS].bool(),
        )
        cell_tokens = self.cell_projection(
            torch.stack(features, dim=-1).to(self.cls_token.dtype)
        )
        cell_tokens = (
            cell_tokens
            + self.row_embedding(self.row_ids)[None]
            + self.col_embedding(self.col_ids)[None]
        )
        cls = self.cls_token.expand(board.shape[0], -1, -1)
        cls = cls + self.empty_count_embedding(empty.sum(dim=1))[:, None]
        encoded = self.transformer(torch.cat((cls, cell_tokens), dim=1))
        return encoded[:, 0], encoded[:, 1:]


class DVN(nn.Module):
    """Supervised decomposed value network predicting final board and winner."""

    def __init__(self, config: ReversiTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or ReversiTransformerConfig()
        self.encoder = BoardEncoder(self.config)
        self.final_board_head = nn.Linear(self.config.d_model, 3)
        self.value_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.GELU(),
            nn.Linear(self.config.d_model, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        board: Tensor,
        current_player: Tensor,
        legal_action_mask: Tensor | None = None,
    ) -> DVNOutput:
        cls, cells = self.encoder(board, current_player, legal_action_mask)
        return DVNOutput(
            final_board_logits=self.final_board_head(cells),
            value=self.value_head(cls).squeeze(-1),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class DPN(nn.Module):
    """Independent policy network trained from frozen-DVN temporal deltas."""

    def __init__(self, config: ReversiTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or ReversiTransformerConfig()
        self.encoder = BoardEncoder(self.config)
        self.cell_head = nn.Linear(self.config.d_model, 1)
        self.pass_head = nn.Linear(self.config.d_model, 1)

    def forward(
        self,
        board: Tensor,
        current_player: Tensor,
        legal_action_mask: Tensor,
    ) -> DPNOutput:
        cls, cells = self.encoder(board, current_player, legal_action_mask)
        logits = torch.cat(
            (self.cell_head(cells).squeeze(-1), self.pass_head(cls)), dim=1
        )
        return DPNOutput(logits.masked_fill(~legal_action_mask.bool(), -torch.inf))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _validate_position_inputs(
    board: Tensor,
    current_player: Tensor,
    legal_action_mask: Tensor | None,
) -> None:
    if board.ndim != 2 or board.shape[1] != BOARD_CELLS:
        raise ValueError("board must have shape [batch, 64]")
    if current_player.shape != (board.shape[0],):
        raise ValueError("current_player must have shape [batch]")
    if legal_action_mask is not None and legal_action_mask.shape != (
        board.shape[0], BOARD_CELLS + 1
    ):
        raise ValueError("legal_action_mask must have shape [batch, 65]")
    # Value-content checks deliberately stay out of this hot path. Reducing
    # torch.all(...) to a Python bool would synchronize CUDA every forward.


def final_board_classes(final_board: Tensor, players: Tensor) -> Tensor:
    """Convert absolute final colors to own/opponent/empty class indices."""
    if final_board.ndim != 2 or final_board.shape[1] != BOARD_CELLS:
        raise ValueError("final_board must have shape [batch, 64]")
    if players.shape != (final_board.shape[0],):
        raise ValueError("players must have shape [batch]")
    player = players[:, None]
    result = torch.full_like(final_board, FINAL_EMPTY, dtype=torch.long)
    result[final_board == player] = FINAL_OWN
    result[final_board == -player] = FINAL_OPPONENT
    return result


def expected_disc_difference(final_board_logits: Tensor) -> Tensor:
    probabilities = final_board_logits.softmax(dim=-1)
    per_cell = probabilities[..., FINAL_OWN] - probabilities[..., FINAL_OPPONENT]
    return per_cell.mean(dim=1)


def dvn_loss(
    output: DVNOutput,
    final_board_target: Tensor,
    value_target: Tensor,
    *,
    board_weight: float = 1.0,
    value_weight: float = 0.5,
) -> dict[str, Tensor]:
    board_loss = F.cross_entropy(
        output.final_board_logits.reshape(-1, 3), final_board_target.reshape(-1)
    )
    value_loss = F.mse_loss(output.value, value_target.to(output.value.dtype))
    return {
        "board": board_loss,
        "value": value_loss,
        "total": board_weight * board_loss + value_weight * value_loss,
    }


def temporal_deltas(
    values: Tensor,
    next_values: Tensor,
    rewards: Tensor,
    terminated: Tensor,
    gamma: float = 1.0,
) -> Tensor:
    """Negamax TD errors; next value belongs to the opposing current player."""
    continuation = (~terminated.bool()).to(values.dtype)
    return rewards.to(values.dtype) - gamma * continuation * next_values - values


def dpn_loss(
    output: DPNOutput,
    actions: Tensor,
    advantages: Tensor,
    *,
    entropy_weight: float = 0.01,
    normalize_advantages: bool = True,
) -> dict[str, Tensor]:
    advantages = advantages.detach().to(output.policy_logits.dtype)
    if normalize_advantages and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-6
        )
    log_probabilities = F.log_softmax(output.policy_logits, dim=-1)
    chosen_log_probabilities = log_probabilities.gather(
        1, actions.long()[:, None]
    ).squeeze(1)
    policy_loss = -(advantages * chosen_log_probabilities).mean()
    probabilities = log_probabilities.exp()
    # Replacing masked -inf before multiplication also keeps its backward path
    # finite. torch.where around ``0 * -inf`` fixes the forward value only.
    safe_log_probabilities = torch.nan_to_num(log_probabilities, neginf=0.0)
    entropy = -(probabilities * safe_log_probabilities).sum(dim=1).mean()
    return {
        "policy": policy_loss,
        "entropy": entropy,
        "total": policy_loss - entropy_weight * entropy,
    }
