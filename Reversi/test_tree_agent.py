import random
import tempfile
import unittest
from pathlib import Path

import torch

from Reversi import PASS_ACTION, ReversiEnv
from Reversi.linear_value import (
    LinearWinPredictor,
    linear_value_loss,
    load_linear_weights,
    save_linear_evaluator,
)
from Reversi.tree_agent import (
    LinearEvaluator,
    TreeSearchAgent,
    apply_move_bits,
    bits_to_actions,
    board_to_bits,
    legal_moves_bits,
)


class BitboardRulesTest(unittest.TestCase):
    def test_random_games_match_scalar_engine_move_for_move(self) -> None:
        rng = random.Random(17)
        for _ in range(20):
            env = ReversiEnv()
            while not env.terminated:
                own, opponent = board_to_bits(env.board, env.current_player)
                bit_actions = bits_to_actions(legal_moves_bits(own, opponent))
                self.assertEqual(bit_actions, list(env.legal_actions()))
                if not bit_actions:
                    env.step(PASS_ACTION)
                    continue

                action = rng.choice(bit_actions)
                next_own, next_opponent = apply_move_bits(own, opponent, action)
                env.step(action)
                expected_own, expected_opponent = board_to_bits(
                    env.board, -env.current_player
                )
                self.assertEqual(next_own, expected_own)
                self.assertEqual(next_opponent, expected_opponent)

    def test_tree_always_returns_legal_action(self) -> None:
        rng = random.Random(29)
        agent = TreeSearchAgent(LinearEvaluator(), depth=3)
        env = ReversiEnv()
        for _ in range(24):
            legal = env.observe().legal_actions
            chosen = agent.select_action(env.board, env.current_player)
            self.assertIn(chosen, legal)
            env.step(rng.choice(legal))


class LinearValueTest(unittest.TestCase):
    def test_lookup_evaluator_matches_direct_weighted_sum(self) -> None:
        weights = tuple((index - 31.5) / 10 for index in range(64))
        evaluator = LinearEvaluator(weights)
        board = [0] * 64
        board[0] = board[17] = board[63] = 1
        board[5] = board[42] = -1
        own, opponent = board_to_bits(board, 1)
        expected = sum(weight * stone for weight, stone in zip(weights, board))
        self.assertAlmostEqual(evaluator.score(own, opponent), expected)

    def test_color_swap_negates_logit_and_loss_backpropagates(self) -> None:
        model = LinearWinPredictor()
        with torch.no_grad():
            model.linear.weight.copy_(torch.arange(64).reshape(1, 64) / 64)
        board = torch.randint(-1, 2, (5, 64), dtype=torch.float32)
        torch.testing.assert_close(model(-board), -model(board))
        loss = linear_value_loss(model(board), torch.tensor([-1, 0, 1, -1, 1]))
        loss.backward()
        self.assertIsNotNone(model.linear.weight.grad)

    def test_symmetrize_and_json_roundtrip(self) -> None:
        model = LinearWinPredictor()
        with torch.no_grad():
            model.linear.weight.copy_(torch.arange(64).reshape(1, 64))
        model.symmetrize_()
        grid = model.linear.weight.detach().reshape(8, 8)
        self.assertTrue(torch.equal(grid, torch.rot90(grid, 1, (0, 1))))
        self.assertTrue(torch.equal(grid, torch.flip(grid, (1,))))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            save_linear_evaluator(model, path, {"test": True})
            self.assertEqual(load_linear_weights(path), model.weights())


if __name__ == "__main__":
    unittest.main()
