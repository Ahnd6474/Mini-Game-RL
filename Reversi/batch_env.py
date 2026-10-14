"""Vectorized NumPy Reversi environments for parallel self-play.

All games live in contiguous arrays. Direction scans loop over the eight board
directions, while every game in the batch is processed simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .env import BLACK, BOARD_CELLS, BOARD_SIZE, PASS_ACTION, WHITE

IntArray = NDArray[np.signedinteger]
BoolArray = NDArray[np.bool_]

_DIRECTIONS = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1),
    (0, 1), (1, -1), (1, 0), (1, 1),
)


@dataclass(slots=True)
class BatchObservation:
    boards: NDArray[np.int8]
    current_players: NDArray[np.int8]
    legal_action_masks: BoolArray
    turns: NDArray[np.int16]
    terminated: BoolArray


@dataclass(slots=True)
class BatchStep:
    actor_rewards: NDArray[np.int8]
    newly_terminated: BoolArray
    winners: NDArray[np.int8]


def _shift(mask: BoolArray, row_delta: int, col_delta: int) -> BoolArray:
    """Shift [N, 8, 8] masks without wrapping around board edges."""
    shifted = np.zeros_like(mask)
    if row_delta >= 0:
        source_rows = slice(0, BOARD_SIZE - row_delta)
        target_rows = slice(row_delta, BOARD_SIZE)
    else:
        source_rows = slice(-row_delta, BOARD_SIZE)
        target_rows = slice(0, BOARD_SIZE + row_delta)
    if col_delta >= 0:
        source_cols = slice(0, BOARD_SIZE - col_delta)
        target_cols = slice(col_delta, BOARD_SIZE)
    else:
        source_cols = slice(-col_delta, BOARD_SIZE)
        target_cols = slice(0, BOARD_SIZE + col_delta)
    shifted[:, target_rows, target_cols] = mask[:, source_rows, source_cols]
    return shifted


class BatchReversiEnv:
    """Many independent Reversi games advanced by one batched ``step`` call."""

    def __init__(self, num_envs: int) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.boards = np.zeros((num_envs, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_players = np.full(num_envs, BLACK, dtype=np.int8)
        self.turns = np.ones(num_envs, dtype=np.int16)
        self.terminated = np.zeros(num_envs, dtype=np.bool_)
        self.consecutive_passes = np.zeros(num_envs, dtype=np.int8)
        self._legal_cache: BoolArray | None = None
        self.reset()

    def reset(self) -> BatchObservation:
        self.boards.fill(0)
        self.boards[:, 3, 3] = WHITE
        self.boards[:, 3, 4] = BLACK
        self.boards[:, 4, 3] = BLACK
        self.boards[:, 4, 4] = WHITE
        self.current_players.fill(BLACK)
        self.turns.fill(1)
        self.terminated.fill(False)
        self.consecutive_passes.fill(0)
        self._legal_cache = None
        return self.observe()

    @property
    def active(self) -> BoolArray:
        return ~self.terminated

    def legal_action_masks(self, *, copy: bool = True) -> BoolArray:
        """Return [N, 65] masks, with pass legal only without placements."""
        if self._legal_cache is None:
            players = self.current_players[:, None, None]
            own = self.boards == players
            opponent = self.boards == -players
            empty = self.boards == 0
            placements = np.zeros_like(empty)
            for row_delta, col_delta in _DIRECTIONS:
                captured = _shift(own, row_delta, col_delta) & opponent
                for _ in range(5):
                    captured |= _shift(captured, row_delta, col_delta) & opponent
                placements |= _shift(captured, row_delta, col_delta) & empty
            placements &= self.active[:, None, None]
            masks = np.zeros((self.num_envs, BOARD_CELLS + 1), dtype=np.bool_)
            masks[:, :BOARD_CELLS] = placements.reshape(self.num_envs, BOARD_CELLS)
            masks[:, PASS_ACTION] = self.active & ~placements.any(axis=(1, 2))
            self._legal_cache = masks
        return self._legal_cache.copy() if copy else self._legal_cache

    def observe(self, *, copy: bool = True) -> BatchObservation:
        def maybe_copy(array: NDArray) -> NDArray:
            return array.copy() if copy else array

        return BatchObservation(
            boards=maybe_copy(self.boards.reshape(self.num_envs, BOARD_CELLS)),
            current_players=maybe_copy(self.current_players),
            legal_action_masks=self.legal_action_masks(copy=copy),
            turns=maybe_copy(self.turns),
            terminated=maybe_copy(self.terminated),
        )

    def step(self, actions: IntArray) -> BatchStep:
        """Advance non-terminated games; terminated-row actions are ignored."""
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs,):
            raise ValueError(f"actions must have shape ({self.num_envs},)")
        if not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("actions must be integers")

        active = self.active
        actor_players = self.current_players.copy()
        legal = self.legal_action_masks(copy=False)
        in_range = (actions >= 0) & (actions <= PASS_ACTION)
        safe_actions = np.clip(actions, 0, PASS_ACTION)
        selected_legal = legal[np.arange(self.num_envs), safe_actions]
        invalid = active & (~in_range | ~selected_legal)
        if invalid.any():
            raise ValueError(
                f"illegal actions for batch rows: {np.flatnonzero(invalid).tolist()}"
            )

        placements = active & (actions < PASS_ACTION)
        passes = active & (actions == PASS_ACTION)
        placement_games = np.flatnonzero(placements)
        if placement_games.size:
            self._apply_placements(placement_games, actions[placement_games])

        self.consecutive_passes[placements] = 0
        self.consecutive_passes[passes] += 1
        board_full = ~(self.boards == 0).any(axis=(1, 2))
        newly_terminated = active & (board_full | (self.consecutive_passes >= 2))
        self.terminated |= newly_terminated
        self.current_players[active] *= -1
        self.turns[active] += 1
        self._legal_cache = None

        disc_differences = self.boards.sum(axis=(1, 2), dtype=np.int16)
        winners = np.zeros(self.num_envs, dtype=np.int8)
        winners[self.terminated] = np.sign(
            disc_differences[self.terminated]
        ).astype(np.int8)
        actor_rewards = np.zeros(self.num_envs, dtype=np.int8)
        actor_rewards[newly_terminated] = (
            winners[newly_terminated] * actor_players[newly_terminated]
        )
        return BatchStep(actor_rewards, newly_terminated, winners)

    def _apply_placements(self, game_ids: IntArray, actions: IntArray) -> None:
        count = game_ids.size
        start_rows = actions // BOARD_SIZE
        start_cols = actions % BOARD_SIZE
        players = self.current_players[game_ids]
        all_flips = np.zeros((count, BOARD_SIZE, BOARD_SIZE), dtype=np.bool_)
        local_ids = np.arange(count)

        for row_delta, col_delta in _DIRECTIONS:
            rows = start_rows.copy()
            cols = start_cols.copy()
            alive = np.ones(count, dtype=np.bool_)
            seen_opponent = np.zeros(count, dtype=np.bool_)
            path = np.zeros_like(all_flips)
            for _ in range(BOARD_SIZE - 1):
                rows += row_delta
                cols += col_delta
                inside = (
                    (rows >= 0) & (rows < BOARD_SIZE)
                    & (cols >= 0) & (cols < BOARD_SIZE)
                )
                scan = alive & inside
                safe_rows = np.clip(rows, 0, BOARD_SIZE - 1)
                safe_cols = np.clip(cols, 0, BOARD_SIZE - 1)
                cells = self.boards[game_ids, safe_rows, safe_cols]
                is_opponent = scan & (cells == -players)
                path[
                    local_ids[is_opponent],
                    safe_rows[is_opponent],
                    safe_cols[is_opponent],
                ] = True
                bracketed = scan & seen_opponent & (cells == players)
                all_flips[bracketed] |= path[bracketed]
                seen_opponent |= is_opponent
                alive = is_opponent
                if not alive.any():
                    break

        self.boards[game_ids, start_rows, start_cols] = players
        self.boards[game_ids] = np.where(
            all_flips,
            players[:, None, None],
            self.boards[game_ids],
        )

    def winners(self) -> NDArray[np.int8]:
        differences = self.boards.sum(axis=(1, 2), dtype=np.int16)
        result = np.zeros(self.num_envs, dtype=np.int8)
        result[self.terminated] = np.sign(
            differences[self.terminated]
        ).astype(np.int8)
        return result
