"""A minimal turn-based Reversi environment.

The environment deliberately has no dependency on Gymnasium or NumPy.  This
makes the rule engine usable by training code, evaluation scripts, and the
interactive submission program without needing adapters first.

Board values:
    EMPTY = 0, BLACK = 1, WHITE = -1

Actions:
    0..63 are cells in row-major order and 64 is PASS_ACTION.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

BOARD_SIZE = 8
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE
PASS_ACTION = BOARD_CELLS

EMPTY = 0
BLACK = 1
WHITE = -1

_DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
_CHAR_TO_STONE = {".": EMPTY, "B": BLACK, "W": WHITE}
_STONE_TO_CHAR = {value: key for key, value in _CHAR_TO_STONE.items()}


class InvalidAction(ValueError):
    """Raised when an agent submits an illegal move or pass."""


@dataclass(frozen=True, slots=True)
class ReversiObservation:
    """Immutable state handed to an agent."""

    board: tuple[int, ...]
    current_player: int
    legal_actions: tuple[int, ...]
    legal_action_mask: tuple[int, ...]
    turn: int
    terminated: bool


def coord_to_action(row: int, col: int) -> int:
    """Convert a problem-statement coordinate (1..8, 1..8) to an action."""
    if not (1 <= row <= BOARD_SIZE and 1 <= col <= BOARD_SIZE):
        raise ValueError(f"coordinate out of range: ({row}, {col})")
    return (row - 1) * BOARD_SIZE + (col - 1)


def action_to_coord(action: int) -> tuple[int, int]:
    """Convert a placement action to a 1-based (row, column) coordinate."""
    if not (0 <= action < BOARD_CELLS):
        raise ValueError(f"not a placement action: {action}")
    return divmod(action, BOARD_SIZE)[0] + 1, action % BOARD_SIZE + 1


class ReversiEnv:
    """Two-player, alternating-turn Reversi environment.

    ``step`` returns ``(observation, rewards, terminated, info)``.  Rewards are
    a dictionary keyed by BLACK (1) and WHITE (-1).  They are zero until the
    game ends, then contain +1 for the winner and -1 for the loser.
    """

    def __init__(
        self,
        board: str | Sequence[int] | None = None,
        current_player: int = BLACK,
        turn: int = 1,
    ) -> None:
        self._board: list[int] = []
        self.current_player = BLACK
        self.turn = 1
        self.terminated = False
        self._consecutive_passes = 0
        self.reset(board=board, current_player=current_player, turn=turn)

    def reset(
        self,
        board: str | Sequence[int] | None = None,
        current_player: int = BLACK,
        turn: int = 1,
    ) -> ReversiObservation:
        """Start from the standard initial position or a supplied position."""
        self._validate_player(current_player)
        if turn < 1:
            raise ValueError("turn must be at least 1")

        if board is None:
            cells = [EMPTY] * BOARD_CELLS
            cells[coord_to_action(4, 4)] = WHITE
            cells[coord_to_action(4, 5)] = BLACK
            cells[coord_to_action(5, 4)] = BLACK
            cells[coord_to_action(5, 5)] = WHITE
        elif isinstance(board, str):
            if len(board) != BOARD_CELLS:
                raise ValueError("board string must contain exactly 64 characters")
            try:
                cells = [_CHAR_TO_STONE[char] for char in board]
            except KeyError as error:
                raise ValueError(f"invalid board character: {error.args[0]!r}") from None
        else:
            cells = list(board)
            if len(cells) != BOARD_CELLS:
                raise ValueError("board must contain exactly 64 cells")
            if any(cell not in (EMPTY, BLACK, WHITE) for cell in cells):
                raise ValueError("board cells must be EMPTY, BLACK, or WHITE")

        self._board = cells
        self.current_player = current_player
        self.turn = turn
        self.terminated = False
        self._consecutive_passes = 0
        return self.observe()

    @property
    def board(self) -> tuple[int, ...]:
        return tuple(self._board)

    def observe(self) -> ReversiObservation:
        legal = () if self.terminated else self.legal_actions()
        mask = [0] * (BOARD_CELLS + 1)
        if not self.terminated:
            if legal:
                for action in legal:
                    mask[action] = 1
            else:
                mask[PASS_ACTION] = 1
        return ReversiObservation(
            board=tuple(self._board),
            current_player=self.current_player,
            legal_actions=legal if legal else (() if self.terminated else (PASS_ACTION,)),
            legal_action_mask=tuple(mask),
            turn=self.turn,
            terminated=self.terminated,
        )

    def legal_actions(self, player: int | None = None) -> tuple[int, ...]:
        """Return legal placement actions. PASS_ACTION is never included here."""
        player = self.current_player if player is None else player
        self._validate_player(player)
        return tuple(
            action
            for action, cell in enumerate(self._board)
            if cell == EMPTY and self._flips_for_action(action, player)
        )

    def step(
        self, action: int
    ) -> tuple[ReversiObservation, dict[int, int], bool, dict[str, object]]:
        """Apply one placement or mandatory pass."""
        if self.terminated:
            raise RuntimeError("cannot step a terminated game; call reset()")
        if not isinstance(action, int) or isinstance(action, bool):
            raise InvalidAction("action must be an integer")

        player = self.current_player
        legal = self.legal_actions(player)
        flipped: tuple[int, ...] = ()
        passed = action == PASS_ACTION

        if legal:
            if passed:
                raise InvalidAction("passing is illegal while a placement exists")
            if action not in legal:
                raise InvalidAction(f"illegal placement action: {action}")
            flipped = self._flips_for_action(action, player)
            self._board[action] = player
            for cell in flipped:
                self._board[cell] = player
            self._consecutive_passes = 0
        else:
            if not passed:
                raise InvalidAction("PASS_ACTION is required when no placement exists")
            self._consecutive_passes += 1

        board_full = EMPTY not in self._board
        self.terminated = board_full or self._consecutive_passes >= 2
        self.current_player = -player
        self.turn += 1

        winner = self.winner() if self.terminated else EMPTY
        rewards = {
            BLACK: 1 if winner == BLACK else (-1 if winner == WHITE else 0),
            WHITE: 1 if winner == WHITE else (-1 if winner == BLACK else 0),
        }
        black_count, white_count = self.disc_counts()
        info: dict[str, object] = {
            "player": player,
            "passed": passed,
            "placed_action": None if passed else action,
            "flipped": flipped,
            "winner": winner,
            "disc_counts": {BLACK: black_count, WHITE: white_count},
        }
        return self.observe(), rewards, self.terminated, info

    def disc_counts(self) -> tuple[int, int]:
        """Return ``(black_count, white_count)``."""
        return self._board.count(BLACK), self._board.count(WHITE)

    def winner(self) -> int:
        """Return BLACK, WHITE, or EMPTY for a draw/not-yet-decided game."""
        if not self.terminated:
            return EMPTY
        black, white = self.disc_counts()
        return BLACK if black > white else WHITE if white > black else EMPTY

    def board_string(self) -> str:
        """Return the 64-character board format used by the judge."""
        return "".join(_STONE_TO_CHAR[cell] for cell in self._board)

    def render(self) -> str:
        """Return a small human-readable board with 1-based coordinates."""
        rows = ["  1 2 3 4 5 6 7 8"]
        board = self.board_string()
        for row in range(BOARD_SIZE):
            start = row * BOARD_SIZE
            rows.append(f"{row + 1} " + " ".join(board[start : start + BOARD_SIZE]))
        player = "B" if self.current_player == BLACK else "W"
        rows.append(f"turn={self.turn} player={player}")
        return "\n".join(rows)

    def clone(self) -> "ReversiEnv":
        """Return an independent copy, useful when an agent searches moves."""
        clone = ReversiEnv(self._board, self.current_player, self.turn)
        clone.terminated = self.terminated
        clone._consecutive_passes = self._consecutive_passes
        return clone

    def _flips_for_action(self, action: int, player: int) -> tuple[int, ...]:
        if not (0 <= action < BOARD_CELLS) or self._board[action] != EMPTY:
            return ()

        row, col = divmod(action, BOARD_SIZE)
        all_flips: list[int] = []
        opponent = -player
        for row_delta, col_delta in _DIRECTIONS:
            scan_row = row + row_delta
            scan_col = col + col_delta
            direction_flips: list[int] = []
            while 0 <= scan_row < BOARD_SIZE and 0 <= scan_col < BOARD_SIZE:
                cell = scan_row * BOARD_SIZE + scan_col
                if self._board[cell] != opponent:
                    if self._board[cell] == player and direction_flips:
                        all_flips.extend(direction_flips)
                    break
                direction_flips.append(cell)
                scan_row += row_delta
                scan_col += col_delta
        return tuple(all_flips)

    @staticmethod
    def _validate_player(player: int) -> None:
        if player not in (BLACK, WHITE):
            raise ValueError("player must be BLACK or WHITE")


def play_episode(agents: dict[int, object]) -> tuple[int, ReversiEnv]:
    """Run one episode for objects exposing ``select_action(observation)``.

    This convenience helper intentionally stays tiny.  Training code can build
    richer replay collection around the same ``observe``/``step`` API.
    """
    if set(agents) != {BLACK, WHITE}:
        raise ValueError("agents must contain exactly BLACK and WHITE")
    env = ReversiEnv()
    while not env.terminated:
        observation = env.observe()
        agent = agents[observation.current_player]
        selector = getattr(agent, "select_action", None)
        if selector is None:
            raise TypeError("each agent must define select_action(observation)")
        env.step(selector(observation))
    return env.winner(), env
