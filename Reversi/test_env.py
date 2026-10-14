import unittest

from Reversi import (
    BLACK,
    EMPTY,
    PASS_ACTION,
    WHITE,
    InvalidAction,
    ReversiEnv,
    coord_to_action,
)


class ReversiEnvTest(unittest.TestCase):
    def test_initial_position_and_legal_actions(self) -> None:
        env = ReversiEnv()
        self.assertEqual(env.disc_counts(), (2, 2))
        self.assertEqual(
            env.legal_actions(),
            tuple(coord_to_action(*coord) for coord in ((3, 4), (4, 3), (5, 6), (6, 5))),
        )

    def test_placement_flips_stone_and_changes_turn(self) -> None:
        env = ReversiEnv()
        observation, rewards, terminated, info = env.step(coord_to_action(3, 4))
        self.assertEqual(env.disc_counts(), (4, 1))
        self.assertEqual(observation.current_player, WHITE)
        self.assertEqual(info["flipped"], (coord_to_action(4, 4),))
        self.assertEqual(rewards, {BLACK: 0, WHITE: 0})
        self.assertFalse(terminated)

    def test_pass_is_only_legal_without_a_placement(self) -> None:
        env = ReversiEnv()
        with self.assertRaises(InvalidAction):
            env.step(PASS_ACTION)

        env.reset("B" * 63 + ".", current_player=WHITE)
        observation, _, terminated, _ = env.step(PASS_ACTION)
        self.assertEqual(observation.legal_actions, (PASS_ACTION,))
        self.assertFalse(terminated)
        _, rewards, terminated, _ = env.step(PASS_ACTION)
        self.assertTrue(terminated)
        self.assertEqual(rewards, {BLACK: 1, WHITE: -1})

    def test_full_board_finishes_immediately(self) -> None:
        board = [BLACK] * 64
        board[coord_to_action(1, 1)] = EMPTY
        board[coord_to_action(1, 2)] = WHITE
        env = ReversiEnv(board, current_player=BLACK)
        _, rewards, terminated, info = env.step(coord_to_action(1, 1))
        self.assertTrue(terminated)
        self.assertEqual(info["flipped"], (coord_to_action(1, 2),))
        self.assertEqual(rewards[BLACK], 1)

    def test_board_string_round_trip_and_clone(self) -> None:
        original = ReversiEnv()
        restored = ReversiEnv(original.board_string())
        clone = restored.clone()
        clone.step(coord_to_action(3, 4))
        self.assertEqual(original.board_string(), restored.board_string())
        self.assertNotEqual(clone.board_string(), restored.board_string())


if __name__ == "__main__":
    unittest.main()
