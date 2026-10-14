import unittest

from Reversi import PASS_ACTION, coord_to_action
from Reversi.symmetry import augment_grid, transform_action, transform_action_values
from Reversi.training_data import TrainingSample, augment_sample


class SymmetryTest(unittest.TestCase):
    def test_four_rotations_return_to_origin(self) -> None:
        action = coord_to_action(2, 3)
        for _ in range(4):
            action = transform_action(action, 1)
        self.assertEqual(action, coord_to_action(2, 3))

    def test_corner_orbit_and_pass(self) -> None:
        corner = coord_to_action(1, 1)
        rotations = {transform_action(corner, symmetry) for symmetry in range(4)}
        self.assertEqual(
            rotations,
            {
                coord_to_action(1, 1),
                coord_to_action(1, 8),
                coord_to_action(8, 1),
                coord_to_action(8, 8),
            },
        )
        self.assertTrue(
            all(transform_action(PASS_ACTION, symmetry) == PASS_ACTION for symmetry in range(8))
        )

    def test_augmentation_transforms_every_spatial_target(self) -> None:
        board = tuple(range(64))
        mask = tuple(range(65))
        sample = TrainingSample(board, 1, mask, board, 1, tuple(float(x) for x in mask))
        augmented = augment_sample(sample)
        self.assertEqual(len(augmented), 8)
        self.assertEqual(tuple(item.board for item in augmented), augment_grid(board))
        for item in augmented:
            self.assertEqual(item.legal_action_mask[PASS_ACTION], PASS_ACTION)
            self.assertEqual(item.policy_target[PASS_ACTION], float(PASS_ACTION))

    def test_action_values_follow_actions(self) -> None:
        values = [0] * 65
        values[coord_to_action(1, 1)] = 1
        values[PASS_ACTION] = 2
        transformed = transform_action_values(values, 1)
        self.assertEqual(transformed[coord_to_action(1, 8)], 1)
        self.assertEqual(transformed[PASS_ACTION], 2)


if __name__ == "__main__":
    unittest.main()
