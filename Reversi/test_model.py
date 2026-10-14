import unittest

import torch

from Reversi import BLACK, ReversiEnv, coord_to_action
from Reversi.model import (
    DPN, DVN, FINAL_EMPTY, FINAL_OPPONENT, FINAL_OWN, dpn_loss, dvn_loss,
    expected_disc_difference, final_board_classes, temporal_deltas,
)


class ReversiModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        observation = ReversiEnv().observe()
        self.board = torch.tensor([observation.board])
        self.player = torch.tensor([observation.current_player])
        self.legal_mask = torch.tensor([observation.legal_action_mask]).bool()
        self.dvn = DVN()
        self.dpn = DPN()

    def test_independent_models_and_forward_shapes(self) -> None:
        dvn_output = self.dvn(self.board, self.player, self.legal_mask)
        dpn_output = self.dpn(self.board, self.player, self.legal_mask)
        self.assertEqual(tuple(dvn_output.final_board_logits.shape), (1, 64, 3))
        self.assertEqual(tuple(dvn_output.value.shape), (1,))
        self.assertEqual(tuple(dpn_output.policy_logits.shape), (1, 65))
        self.assertTrue(torch.isfinite(dpn_output.policy_logits[self.legal_mask]).all())
        self.assertTrue(torch.isneginf(dpn_output.policy_logits[~self.legal_mask]).all())
        dvn_pointers = {parameter.data_ptr() for parameter in self.dvn.parameters()}
        dpn_pointers = {parameter.data_ptr() for parameter in self.dpn.parameters()}
        self.assertTrue(dvn_pointers.isdisjoint(dpn_pointers))

    def test_final_board_target_is_relative_to_player(self) -> None:
        final_board = torch.tensor([[BLACK, -BLACK, 0] + [0] * 61])
        target = final_board_classes(final_board, torch.tensor([BLACK]))
        self.assertEqual(target[0, :3].tolist(), [FINAL_OWN, FINAL_OPPONENT, FINAL_EMPTY])

    def test_separate_losses_backward(self) -> None:
        dvn_output = self.dvn(self.board, self.player, self.legal_mask)
        final_target = final_board_classes(self.board, self.player)
        value_losses = dvn_loss(dvn_output, final_target, torch.tensor([1.0]))
        value_losses["total"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.dvn.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in self.dpn.parameters()))

        dpn_output = self.dpn(self.board, self.player, self.legal_mask)
        policy_losses = dpn_loss(
            dpn_output,
            torch.tensor([coord_to_action(3, 4)]),
            torch.tensor([0.25]),
            normalize_advantages=False,
        )
        policy_losses["total"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.dpn.parameters()))
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in self.dpn.parameters()
            )
        )

    def test_negamax_temporal_delta(self) -> None:
        delta = temporal_deltas(
            values=torch.tensor([0.1, 0.2]),
            next_values=torch.tensor([-0.4, 0.7]),
            rewards=torch.tensor([0.0, 1.0]),
            terminated=torch.tensor([False, True]),
        )
        self.assertTrue(torch.allclose(delta, torch.tensor([0.3, 0.8])))

    def test_expected_disc_difference_shape(self) -> None:
        output = self.dvn(self.board, self.player, self.legal_mask)
        self.assertEqual(tuple(expected_disc_difference(output.final_board_logits).shape), (1,))


if __name__ == "__main__":
    unittest.main()
