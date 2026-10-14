"""Standalone Reversi submission: bitboard alpha-beta search to depth 5."""

import math
import sys
import time


FULL = (1 << 64) - 1
NOT_A = 0xFEFEFEFEFEFEFEFE
NOT_H = 0x7F7F7F7F7F7F7F7F
PASS = 64
WIN_SCORE = 10_000.0
# Leave margin for parsing, output and interpreter overhead under the 5 s limit.
CPU_DEADLINE = time.process_time() + 4.65

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

# Learned from 990,170 random-play positions, then D4-symmetrized.
WEIGHTS = (
    0.7631090879440308, 0.004666804801672697, 0.07128506898880005,
    0.049976978451013565, 0.049976978451013565, 0.07128506898880005,
    0.004666804801672697, 0.7631090879440308, 0.004666804801672697,
    -0.04120081290602684, 0.008824602700769901, 0.02624380774796009,
    0.02624380774796009, 0.008824602700769901, -0.04120081290602684,
    0.004666804801672697, 0.07128506898880005, 0.008824602700769901,
    0.028687071055173874, 0.001383814960718155, 0.001383814960718155,
    0.028687071055173874, 0.008824602700769901, 0.07128506898880005,
    0.049976978451013565, 0.02624380774796009, 0.001383814960718155,
    0.002079043071717024, 0.002079043071717024, 0.001383814960718155,
    0.02624380774796009, 0.049976978451013565, 0.049976978451013565,
    0.02624380774796009, 0.001383814960718155, 0.002079043071717024,
    0.002079043071717024, 0.001383814960718155, 0.02624380774796009,
    0.049976978451013565, 0.07128506898880005, 0.008824602700769901,
    0.028687071055173874, 0.001383814960718155, 0.001383814960718155,
    0.028687071055173874, 0.008824602700769901, 0.07128506898880005,
    0.004666804801672697, -0.04120081290602684, 0.008824602700769901,
    0.02624380774796009, 0.02624380774796009, 0.008824602700769901,
    -0.04120081290602684, 0.004666804801672697, 0.7631090879440308,
    0.004666804801672697, 0.07128506898880005, 0.049976978451013565,
    0.049976978451013565, 0.07128506898880005, 0.004666804801672697,
    0.7631090879440308,
)


def _make_byte_tables():
    tables = []
    for byte_index in range(8):
        offset = byte_index * 8
        tables.append(tuple(
            sum(WEIGHTS[offset + bit] for bit in range(8) if pattern & (1 << bit))
            for pattern in range(256)
        ))
    return tuple(tables)


BYTE_TABLES = _make_byte_tables()


def shift(bits, direction):
    if direction == 0:
        return bits >> 8
    if direction == 1:
        return (bits << 8) & FULL
    if direction == 2:
        return ((bits & NOT_H) << 1) & FULL
    if direction == 3:
        return (bits & NOT_A) >> 1
    if direction == 4:
        return (bits & NOT_H) >> 7
    if direction == 5:
        return (bits & NOT_A) >> 9
    if direction == 6:
        return ((bits & NOT_H) << 9) & FULL
    return ((bits & NOT_A) << 7) & FULL


def legal_moves(player, opponent):
    empty = FULL ^ (player | opponent)
    moves = 0
    for direction in range(8):
        captured = shift(player, direction) & opponent
        for _ in range(5):
            captured |= shift(captured, direction) & opponent
        moves |= shift(captured, direction) & empty
    return moves


def apply_move(player, opponent, action):
    move = 1 << action
    flips = 0
    for direction in range(8):
        cursor = shift(move, direction)
        captured = 0
        while cursor & opponent:
            captured |= cursor
            cursor = shift(cursor, direction)
        if cursor & player:
            flips |= captured
    return player | move | flips, opponent & ~flips


def ordered_actions(moves, preferred=-1):
    actions = []
    while moves:
        bit = moves & -moves
        actions.append(bit.bit_length() - 1)
        moves ^= bit
    actions.sort(
        key=lambda action: (
            action == preferred,
            action in (0, 7, 56, 63),
            POSITION_ORDER[action],
            WEIGHTS[action],
        ),
        reverse=True,
    )
    return actions


def leaf_score(player, opponent):
    return sum(
        BYTE_TABLES[index][(player >> (index * 8)) & 255]
        - BYTE_TABLES[index][(opponent >> (index * 8)) & 255]
        for index in range(8)
    )


class SearchTimeout(Exception):
    pass


class Search:
    __slots__ = ("table", "nodes")

    def __init__(self):
        self.table = {}
        self.nodes = 0

    def negamax(self, player, opponent, depth, alpha, beta, passed):
        self.nodes += 1
        if self.nodes & 511 == 0 and time.process_time() >= CPU_DEADLINE:
            raise SearchTimeout
        key = (player, opponent, depth, passed)
        alpha_original = alpha
        beta_original = beta
        entry = self.table.get(key)
        preferred = -1
        if entry is not None:
            value, flag, preferred = entry
            if flag == 0:
                return value
            if flag > 0:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                return value

        moves = legal_moves(player, opponent)
        if not moves:
            if passed:
                difference = player.bit_count() - opponent.bit_count()
                if difference == 0:
                    return 0.0
                return math.copysign(WIN_SCORE + abs(difference), difference)
            return -self.negamax(opponent, player, depth, -beta, -alpha, True)
        if depth == 0:
            return leaf_score(player, opponent)

        best = -math.inf
        best_action = -1
        for action in ordered_actions(moves, preferred):
            next_player, next_opponent = apply_move(player, opponent, action)
            value = -self.negamax(
                next_opponent, next_player, depth - 1, -beta, -alpha, False
            )
            if value > best:
                best = value
                best_action = action
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        flag = -1 if best <= alpha_original else (1 if best >= beta_original else 0)
        self.table[key] = (best, flag, best_action)
        return best

    def root(self, player, opponent, depth, fallback):
        moves = legal_moves(player, opponent)
        alpha = -math.inf
        best_action = fallback
        for action in ordered_actions(moves, fallback):
            if time.process_time() >= CPU_DEADLINE:
                raise SearchTimeout
            next_player, next_opponent = apply_move(player, opponent, action)
            value = -self.negamax(
                next_opponent, next_player, depth - 1, -math.inf, -alpha, False
            )
            if value > alpha:
                alpha = value
                best_action = action
        return best_action


def parse_bits(board, color):
    own = 0
    opponent = 0
    own_char = "B" if color == 1 else "W"
    opponent_char = "W" if color == 1 else "B"
    for index, cell in enumerate(board):
        if cell == own_char:
            own |= 1 << index
        elif cell == opponent_char:
            opponent |= 1 << index
    return own, opponent


def select_action(board, color):
    own, opponent = parse_bits(board, color)
    moves = legal_moves(own, opponent)
    if not moves:
        return PASS
    fallback = ordered_actions(moves)[0]
    if time.process_time() >= CPU_DEADLINE:
        return fallback
    search = Search()
    # Depth 4 guarantees a stable answer if the depth-5 pass hits the safety cap.
    for depth in (4, 5):
        try:
            fallback = search.root(own, opponent, depth, fallback)
        except SearchTimeout:
            break
    return fallback


def main():
    color = 1
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "READY":
            color = 1 if parts[1] == "FIRST" else -1
            print("OK", flush=True)
        elif parts[0] == "TURN":
            action = select_action(parts[2], color)
            if action == PASS:
                print("PASS", flush=True)
            else:
                print("MOVE", action // 8 + 1, action % 8 + 1, flush=True)


if __name__ == "__main__":
    main()
