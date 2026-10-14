"""The eight rotation/reflection symmetries of an 8x8 Reversi board."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from .env import BOARD_CELLS, BOARD_SIZE, PASS_ACTION

T = TypeVar("T")

SYMMETRY_NAMES = (
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "mirror",
    "mirror_rotate_90",
    "mirror_rotate_180",
    "mirror_rotate_270",
)


def transform_coord(row: int, col: int, symmetry: int) -> tuple[int, int]:
    """Transform a zero-based board coordinate with one D4 symmetry."""
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError("row and col must be in 0..7")
    if not (0 <= symmetry < len(SYMMETRY_NAMES)):
        raise ValueError("symmetry must be in 0..7")

    rotations = symmetry % 4
    if symmetry >= 4:
        col = BOARD_SIZE - 1 - col
    for _ in range(rotations):
        row, col = col, BOARD_SIZE - 1 - row
    return row, col


def transform_action(action: int, symmetry: int) -> int:
    """Transform a placement action; pass remains pass."""
    if action == PASS_ACTION:
        return PASS_ACTION
    if not (0 <= action < BOARD_CELLS):
        raise ValueError("action must be in 0..64")
    row, col = divmod(action, BOARD_SIZE)
    new_row, new_col = transform_coord(row, col, symmetry)
    return new_row * BOARD_SIZE + new_col


def transform_grid(values: Sequence[T], symmetry: int) -> tuple[T, ...]:
    """Transform 64 row-major values such as a board or per-cell labels."""
    if len(values) != BOARD_CELLS:
        raise ValueError("grid must contain exactly 64 values")
    transformed: list[T | None] = [None] * BOARD_CELLS
    for old_action, value in enumerate(values):
        transformed[transform_action(old_action, symmetry)] = value
    return tuple(transformed)  # type: ignore[arg-type]


def transform_action_values(values: Sequence[T], symmetry: int) -> tuple[T, ...]:
    """Transform 65 action values/masks while preserving the pass entry."""
    if len(values) != BOARD_CELLS + 1:
        raise ValueError("action values must contain exactly 65 entries")
    return transform_grid(values[:BOARD_CELLS], symmetry) + (values[PASS_ACTION],)


def augment_grid(values: Sequence[T]) -> tuple[tuple[T, ...], ...]:
    """Return all eight symmetric versions of a 64-cell grid."""
    return tuple(transform_grid(values, symmetry) for symmetry in range(8))
