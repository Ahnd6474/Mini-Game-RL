import unittest

from Reversi.env import BLACK, PASS_ACTION, ReversiEnv
from Reversi.submission_tree_depth5 import select_action


def _board_text(board: tuple[int, ...]) -> str:
    return "".join("." if cell == 0 else ("B" if cell == 1 else "W") for cell in board)


class TreeSubmissionTest(unittest.TestCase):
    def test_initial_action_is_legal(self) -> None:
        env = ReversiEnv()
        action = select_action(_board_text(env.board), BLACK)
        self.assertIn(action, env.observe().legal_actions)

    def test_pass_position_returns_pass(self) -> None:
        board = "B" * 63 + "."
        self.assertEqual(select_action(board, -1), PASS_ACTION)


if __name__ == "__main__":
    unittest.main()
