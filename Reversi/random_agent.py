"""A minimal agent and smoke-test match for the Reversi environment."""

from __future__ import annotations

import random

from Reversi import BLACK, WHITE, ReversiObservation
from Reversi.env import play_episode


class RandomAgent:
    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def select_action(self, observation: ReversiObservation) -> int:
        return self._random.choice(observation.legal_actions)


if __name__ == "__main__":
    winner, final_env = play_episode(
        {BLACK: RandomAgent(seed=1), WHITE: RandomAgent(seed=2)}
    )
    print(final_env.render())
    print("winner:", {BLACK: "B", WHITE: "W", 0: "draw"}[winner])
