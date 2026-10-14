"""Fast bitboard negamax/alpha-beta agent with a learned linear leaf value."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .env import BOARD_CELLS, PASS_ACTION, action_to_coord
from .linear_value import load_linear_weights

FULL = (1 << 64) - 1
NOT_A = 0xFEFEFEFEFEFEFEFE
NOT_H = 0x7F7F7F7F7F7F7F7F
CORNER_MASK = (1 << 0) | (1 << 7) | (1 << 56) | (1 << 63)
WIN_SCORE = 10_000.0

# Fixed fallback ordering. Corners and safe edges are searched first.
POSITION_ORDER = (
    120, -25, 20, 5, 5, 20, -25, 120,
    -25, -45, -5, -5, -5, -5, -45, -25,
    20, -5, 15, 3, 3, 15, -5, 20,
    5, -5, 3, 3, 3, 3, -5, 5,
    5, -5, 3, 3, 3, 3, -5, 5,
    20, -5, 15, 3, 3, 15, -5, 20,
    -25, -45, -5, -5, -5, -5, -45, -25,
    120, -25, 20, 5, 5, 20, -25, 120,
)
DEFAULT_LINEAR_WEIGHTS = tuple(value / 30.0 for value in POSITION_ORDER)


def _shift(bits: int, direction: int) -> int:
    if direction == 0:  # north
        return bits >> 8
    if direction == 1:  # south
        return (bits << 8) & FULL
    if direction == 2:  # east
        return ((bits & NOT_H) << 1) & FULL
    if direction == 3:  # west
        return (bits & NOT_A) >> 1
    if direction == 4:  # north-east
        return (bits & NOT_H) >> 7
    if direction == 5:  # north-west
        return (bits & NOT_A) >> 9
    if direction == 6:  # south-east
        return ((bits & NOT_H) << 9) & FULL
    return ((bits & NOT_A) << 7) & FULL  # south-west


def legal_moves_bits(player: int, opponent: int) -> int:
    empty = FULL ^ (player | opponent)
    moves = 0
    for direction in range(8):
        captured = _shift(player, direction) & opponent
        for _ in range(5):
            captured |= _shift(captured, direction) & opponent
        moves |= _shift(captured, direction) & empty
    return moves


def apply_move_bits(player: int, opponent: int, action: int) -> tuple[int, int]:
    move = 1 << action
    if move & (player | opponent):
        raise ValueError("move is not empty")
    flips = 0
    for direction in range(8):
        cursor = _shift(move, direction)
        captured = 0
        while cursor and cursor & opponent:
            captured |= cursor
            cursor = _shift(cursor, direction)
        if cursor & player:
            flips |= captured
    if not flips:
        raise ValueError("move flips no stones")
    return player | move | flips, opponent & ~flips


def board_to_bits(board: Sequence[int], player: int) -> tuple[int, int]:
    if len(board) != BOARD_CELLS:
        raise ValueError("board must contain 64 cells")
    own = 0
    other = 0
    for action, stone in enumerate(board):
        if stone == player:
            own |= 1 << action
        elif stone == -player:
            other |= 1 << action
    return own, other


def bits_to_actions(moves: int) -> list[int]:
    result = []
    while moves:
        bit = moves & -moves
        result.append(bit.bit_length() - 1)
        moves ^= bit
    return result


@dataclass(slots=True)
class SearchStats:
    nodes: int = 0
    cutoffs: int = 0
    transposition_hits: int = 0


@dataclass(slots=True)
class _Entry:
    value: float
    flag: int  # -1 upper bound, 0 exact, 1 lower bound
    best_action: int


class LinearEvaluator:
    __slots__ = ("weights", "_byte_tables")

    def __init__(self, weights: Sequence[float] = DEFAULT_LINEAR_WEIGHTS) -> None:
        if len(weights) != BOARD_CELLS:
            raise ValueError("evaluator needs exactly 64 weights")
        self.weights = tuple(float(value) for value in weights)
        tables: list[tuple[float, ...]] = []
        for byte_index in range(8):
            offset = byte_index * 8
            table = []
            for pattern in range(256):
                table.append(
                    sum(
                        self.weights[offset + bit]
                        for bit in range(8)
                        if pattern & (1 << bit)
                    )
                )
            tables.append(tuple(table))
        self._byte_tables = tuple(tables)

    @classmethod
    def from_json(cls, path: str | Path) -> "LinearEvaluator":
        return cls(load_linear_weights(path))

    def score(self, player: int, opponent: int) -> float:
        tables = self._byte_tables
        return sum(
            tables[index][(player >> (index * 8)) & 255]
            - tables[index][(opponent >> (index * 8)) & 255]
            for index in range(8)
        )

    def probability(self, player: int, opponent: int) -> float:
        score = max(-40.0, min(40.0, self.score(player, opponent)))
        return 1.0 / (1.0 + math.exp(-score))


class TreeSearchAgent:
    """Deterministic fixed-depth search agent suitable as a benchmark anchor."""

    def __init__(self, evaluator: LinearEvaluator, depth: int = 4) -> None:
        if depth < 1:
            raise ValueError("depth must be positive")
        self.evaluator = evaluator
        self.depth = depth
        self.stats = SearchStats()
        self._table: dict[tuple[int, int, int, bool], _Entry] = {}

    def select_action(self, board: Sequence[int], player: int) -> int:
        own, opponent = board_to_bits(board, player)
        moves = legal_moves_bits(own, opponent)
        if not moves:
            return PASS_ACTION
        self.stats = SearchStats()
        self._table.clear()
        alpha = -math.inf
        beta = math.inf
        best_action = -1
        for action in self._ordered_actions(moves, -1):
            next_own, next_opponent = apply_move_bits(own, opponent, action)
            value = -self._negamax(
                next_opponent,
                next_own,
                self.depth - 1,
                -beta,
                -alpha,
                False,
            )
            if value > alpha:
                alpha = value
                best_action = action
        return best_action

    def _negamax(
        self,
        player: int,
        opponent: int,
        depth: int,
        alpha: float,
        beta: float,
        passed: bool,
    ) -> float:
        self.stats.nodes += 1
        key = (player, opponent, depth, passed)
        alpha_original = alpha
        beta_original = beta
        entry = self._table.get(key)
        tt_action = -1
        if entry is not None:
            self.stats.transposition_hits += 1
            tt_action = entry.best_action
            if entry.flag == 0:
                return entry.value
            if entry.flag > 0:
                alpha = max(alpha, entry.value)
            else:
                beta = min(beta, entry.value)
            if alpha >= beta:
                return entry.value

        moves = legal_moves_bits(player, opponent)
        if not moves:
            if passed:
                difference = player.bit_count() - opponent.bit_count()
                return 0.0 if difference == 0 else math.copysign(
                    WIN_SCORE + abs(difference), difference
                )
            return -self._negamax(opponent, player, depth, -beta, -alpha, True)
        if depth == 0:
            return self.evaluator.score(player, opponent)

        best = -math.inf
        best_action = -1
        for action in self._ordered_actions(moves, tt_action):
            next_player, next_opponent = apply_move_bits(player, opponent, action)
            value = -self._negamax(
                next_opponent,
                next_player,
                depth - 1,
                -beta,
                -alpha,
                False,
            )
            if value > best:
                best = value
                best_action = action
            alpha = max(alpha, value)
            if alpha >= beta:
                self.stats.cutoffs += 1
                break

        flag = 0
        if best <= alpha_original:
            flag = -1
        elif best >= beta_original:
            flag = 1
        self._table[key] = _Entry(best, flag, best_action)
        return best

    def _ordered_actions(self, moves: int, tt_action: int) -> list[int]:
        actions = bits_to_actions(moves)
        actions.sort(
            key=lambda action: (
                action == tt_action,
                bool((1 << action) & CORNER_MASK),
                POSITION_ORDER[action],
                self.evaluator.weights[action],
            ),
            reverse=True,
        )
        return actions


def _parse_board(text: str) -> list[int]:
    if len(text) != BOARD_CELLS or any(char not in ".BW" for char in text):
        raise ValueError("board must contain exactly 64 characters from .BW")
    return [0 if char == "." else (1 if char == "B" else -1) for char in text]


def interactive_main(weights: Path | None, depth: int) -> None:
    evaluator = (
        LinearEvaluator.from_json(weights) if weights is not None else LinearEvaluator()
    )
    agent = TreeSearchAgent(evaluator, depth)
    player = 1
    for line in sys.stdin:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "READY":
            player = 1 if parts[1] == "FIRST" else -1
            print("OK", flush=True)
        elif parts[0] == "TURN":
            action = agent.select_action(_parse_board(parts[2]), player)
            if action == PASS_ACTION:
                print("PASS", flush=True)
            else:
                row, col = action_to_coord(action)
                print("MOVE", row, col, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--depth", type=int, default=4)
    args = parser.parse_args()
    interactive_main(args.weights, args.depth)


if __name__ == "__main__":
    main()
