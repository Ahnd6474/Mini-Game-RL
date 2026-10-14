"""Small, dependency-free Reversi environment for agents and self-play."""

from .env import (
    BLACK,
    BOARD_CELLS,
    BOARD_SIZE,
    EMPTY,
    PASS_ACTION,
    WHITE,
    InvalidAction,
    ReversiEnv,
    ReversiObservation,
    action_to_coord,
    coord_to_action,
)

__all__ = [
    "BLACK",
    "BOARD_CELLS",
    "BOARD_SIZE",
    "EMPTY",
    "PASS_ACTION",
    "WHITE",
    "InvalidAction",
    "ReversiEnv",
    "ReversiObservation",
    "action_to_coord",
    "coord_to_action",
]
