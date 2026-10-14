import unittest

from Reversi.model import DPN, DVN, ReversiTransformerConfig
from Reversi.trainer import SelfPlayTrainer, TrainerConfig


class TrainerTest(unittest.TestCase):
    def test_small_end_to_end_iteration(self) -> None:
        model_config = ReversiTransformerConfig(
            d_model=16,
            num_layers=1,
            num_heads=2,
            dim_feedforward=32,
            dropout=0.0,
        )
        trainer_config = TrainerConfig(
            games_per_iteration=2,
            minibatch_size=256,
            dvn_epochs=1,
            dpn_epochs=1,
            device="cpu",
            seed=3,
        )
        trainer = SelfPlayTrainer(
            DVN(model_config), DPN(model_config), trainer_config
        )
        metrics = trainer.train_iteration()
        self.assertEqual(metrics["games"], 2)
        self.assertGreater(metrics["positions"], 100)
        self.assertIn("dvn_total", metrics)
        self.assertIn("dpn_total", metrics)


if __name__ == "__main__":
    unittest.main()
