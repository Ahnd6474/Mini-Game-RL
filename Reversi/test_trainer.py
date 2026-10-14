import unittest

import numpy as np
import torch

from Reversi.model import DPN, DVN, ReversiTransformerConfig
from Reversi.trainer import DVNReplayBuffer, SelfPlayTrainer, TrainerConfig


class TrainerTest(unittest.TestCase):
    def _trainer(self, games: int = 2) -> SelfPlayTrainer:
        model_config = ReversiTransformerConfig(
            d_model=16,
            num_layers=1,
            num_heads=2,
            dim_feedforward=32,
            dropout=0.0,
        )
        trainer_config = TrainerConfig(
            games_per_iteration=games,
            minibatch_size=256,
            dvn_epochs=1,
            dpn_epochs=1,
            device="cpu",
            seed=3,
        )
        return SelfPlayTrainer(
            DVN(model_config), DPN(model_config), trainer_config
        )

    def test_small_end_to_end_iteration(self) -> None:
        trainer = self._trainer()
        metrics = trainer.train_iteration()
        self.assertEqual(metrics["games"], 2)
        self.assertGreater(metrics["positions"], 100)
        self.assertIn("dvn_total", metrics)
        self.assertIn("dpn_total", metrics)
        self.assertIn("dpn_clip_fraction", metrics)
        self.assertIn("dpn_approx_kl", metrics)

    def test_rollout_old_log_probabilities_match_exact_policy_views(self) -> None:
        trainer = self._trainer()
        rollout = trainer.collect_self_play()
        self.assertTrue(
            rollout.policy_legal_masks[
                np.arange(len(rollout)), rollout.actions
            ].all()
        )
        with torch.inference_mode():
            output = trainer.dpn(
                torch.from_numpy(rollout.policy_boards),
                torch.from_numpy(rollout.players),
                torch.from_numpy(rollout.policy_legal_masks),
            )
            actual = torch.log_softmax(
                output.policy_logits.float(), dim=1
            ).gather(1, torch.from_numpy(rollout.actions)[:, None]).squeeze(1)
        torch.testing.assert_close(
            actual, torch.from_numpy(rollout.old_log_probabilities)
        )

    def test_replay_buffer_caps_and_round_trips(self) -> None:
        trainer = self._trainer(games=2)
        rollout = trainer.collect_self_play()
        replay = DVNReplayBuffer(capacity=50, seed=7)
        replay.add(rollout)
        self.assertEqual(len(replay), 50)
        sample = replay.sample(20)
        self.assertIsNotNone(sample)
        self.assertEqual(len(sample), 20)
        restored = DVNReplayBuffer(capacity=30, seed=8)
        restored.load_state_dict(replay.state_dict())
        self.assertEqual(len(restored), 30)
        np.testing.assert_array_equal(
            np.sort(restored.priorities), np.sort(replay.priorities)[-30:]
        )

    def test_pool_opponent_actions_are_excluded_from_ppo(self) -> None:
        model_config = ReversiTransformerConfig(
            d_model=16,
            num_layers=1,
            num_heads=2,
            dim_feedforward=32,
            dropout=0.0,
        )
        trainer = SelfPlayTrainer(
            DVN(model_config),
            DPN(model_config),
            TrainerConfig(
                games_per_iteration=4,
                minibatch_size=256,
                dvn_epochs=1,
                dpn_epochs=1,
                dvn_replay_capacity=0,
                opponent_pool_size=2,
                opponent_pool_fraction=1.0,
                device="cpu",
                seed=11,
            ),
        )
        trainer.add_policy_snapshot()
        rollout = trainer.collect_self_play()
        self.assertEqual(rollout.opponent_pool_games, 4)
        self.assertTrue(rollout.policy_trainable.any())
        self.assertTrue((~rollout.policy_trainable).any())
        metrics = trainer.train_iteration()
        self.assertLess(metrics["dpn_train_positions"], metrics["positions"])
        self.assertEqual(metrics["opponent_pool_games"], 4)


if __name__ == "__main__":
    unittest.main()
