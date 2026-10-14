"""Measure fixed-depth bitboard search on reproducible midgame positions."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .env import ReversiEnv
from .tree_agent import LinearEvaluator, TreeSearchAgent


def make_positions(count: int, plies: int, seed: int) -> list[tuple[tuple[int, ...], int]]:
    rng = random.Random(seed)
    positions: list[tuple[tuple[int, ...], int]] = []
    while len(positions) < count:
        env = ReversiEnv()
        for _ in range(plies):
            if env.terminated:
                break
            env.step(rng.choice(env.observe().legal_actions))
        if not env.terminated and env.legal_actions():
            positions.append((env.board, env.current_player))
    return positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--depths", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--plies", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    evaluator = (
        LinearEvaluator.from_json(args.weights)
        if args.weights is not None
        else LinearEvaluator()
    )
    positions = make_positions(args.positions, args.plies, args.seed)
    for depth in args.depths:
        elapsed = 0.0
        nodes = 0
        cutoffs = 0
        hits = 0
        for board, player in positions:
            agent = TreeSearchAgent(evaluator, depth)
            started = time.perf_counter()
            agent.select_action(board, player)
            elapsed += time.perf_counter() - started
            nodes += agent.stats.nodes
            cutoffs += agent.stats.cutoffs
            hits += agent.stats.transposition_hits
        print(
            json.dumps(
                {
                    "depth": depth,
                    "positions": len(positions),
                    "mean_ms": elapsed * 1000 / len(positions),
                    "total_nodes": nodes,
                    "nodes_per_second": nodes / elapsed,
                    "cutoffs": cutoffs,
                    "transposition_hits": hits,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
