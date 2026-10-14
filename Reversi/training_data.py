"""Training sample representation and symmetry augmentation helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .env import BOARD_CELLS, BLACK, WHITE
from .symmetry import transform_action_values, transform_grid


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """One position labelled with its episode's final result.

    Board colors are absolute (-1/0/1).  Outcome is relative to
    ``current_player``: win=1, draw=0, loss=-1.
    """

    board: tuple[int, ...]
    current_player: int
    legal_action_mask: tuple[int, ...]
    final_board: tuple[int, ...]
    outcome: int
    policy_target: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.board) != BOARD_CELLS or len(self.final_board) != BOARD_CELLS:
            raise ValueError("board and final_board must contain 64 cells")
        if len(self.legal_action_mask) != BOARD_CELLS + 1:
            raise ValueError("legal_action_mask must contain 65 entries")
        if self.current_player not in (BLACK, WHITE):
            raise ValueError("current_player must be BLACK or WHITE")
        if self.outcome not in (-1, 0, 1):
            raise ValueError("outcome must be -1, 0, or 1")
        if self.policy_target is not None and len(self.policy_target) != BOARD_CELLS + 1:
            raise ValueError("policy_target must contain 65 entries")


def augment_sample(sample: TrainingSample) -> tuple[TrainingSample, ...]:
    """Create the eight D4-equivalent versions of a labelled position."""
    augmented = []
    for symmetry in range(8):
        policy = (
            None
            if sample.policy_target is None
            else transform_action_values(sample.policy_target, symmetry)
        )
        augmented.append(
            replace(
                sample,
                board=transform_grid(sample.board, symmetry),
                legal_action_mask=transform_action_values(
                    sample.legal_action_mask, symmetry
                ),
                final_board=transform_grid(sample.final_board, symmetry),
                policy_target=policy,
            )
        )
    return tuple(augmented)
