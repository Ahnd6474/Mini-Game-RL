import random
import unittest

import numpy as np

from Reversi import PASS_ACTION, ReversiEnv
from Reversi.batch_env import BatchReversiEnv


class BatchReversiEnvTest(unittest.TestCase):
    def test_initial_masks_match_scalar_environment(self) -> None:
        batch = BatchReversiEnv(4)
        expected = np.asarray(ReversiEnv().observe().legal_action_mask, dtype=np.bool_)
        self.assertTrue(np.all(batch.legal_action_masks() == expected[None, :]))

    def test_parallel_games_match_scalar_rule_engine(self) -> None:
        count = 16
        rng = random.Random(7)
        batch = BatchReversiEnv(count)
        scalar = [ReversiEnv() for _ in range(count)]
        while batch.active.any():
            masks = batch.legal_action_masks(copy=False)
            actions = np.full(count, PASS_ACTION, dtype=np.int64)
            for game_id, env in enumerate(scalar):
                if env.terminated:
                    continue
                action = rng.choice(np.flatnonzero(masks[game_id]).tolist())
                actions[game_id] = action
                env.step(action)
            batch.step(actions)
            for game_id, env in enumerate(scalar):
                self.assertEqual(
                    batch.boards[game_id].reshape(-1).tolist(), list(env.board)
                )
                self.assertEqual(int(batch.current_players[game_id]), env.current_player)
                self.assertEqual(bool(batch.terminated[game_id]), env.terminated)
        np.testing.assert_array_equal(
            batch.winners(), np.asarray([env.winner() for env in scalar])
        )

    def test_illegal_action_is_rejected(self) -> None:
        batch = BatchReversiEnv(2)
        with self.assertRaises(ValueError):
            batch.step(np.asarray([0, 0]))


if __name__ == "__main__":
    unittest.main()
