import unittest

import numpy as np
import torch

from Reversi.train_linear import collect_random_games, train_linear_model


class LinearBootstrapTest(unittest.TestCase):
    def test_small_collection_and_training(self) -> None:
        dataset = collect_random_games(32, seed=5)
        self.assertEqual(dataset.boards.shape[1], 64)
        self.assertEqual(dataset.outcomes.shape, (len(dataset),))
        self.assertTrue(np.isin(dataset.boards, (-1, 0, 1)).all())
        self.assertTrue(np.isin(dataset.outcomes, (-1, 0, 1)).all())

        model, metrics = train_linear_model(
            dataset,
            device=torch.device("cpu"),
            epochs=1,
            batch_size=512,
            seed=5,
        )
        self.assertEqual(len(model.weights()), 64)
        self.assertEqual(metrics["games"], 32)
        grid = np.asarray(model.weights()).reshape(8, 8)
        np.testing.assert_allclose(grid, np.rot90(grid))


if __name__ == "__main__":
    unittest.main()
