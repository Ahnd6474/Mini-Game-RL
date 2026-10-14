"""Command-line entry point for parallel DVN/DPN self-play training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import DPN, DVN
from .export_submission import export_dpn_submission
from .trainer import SelfPlayTrainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--dvn-epochs", type=int, default=1)
    parser.add_argument("--dpn-epochs", type=int, default=4)
    parser.add_argument("--dpn-learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--ppo-target-kl", type=float, default=0.02)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dvn-replay-capacity", type=int, default=262_144)
    parser.add_argument("--dvn-replay-fraction", type=float, default=0.5)
    parser.add_argument("--opponent-pool-size", type=int, default=4)
    parser.add_argument("--opponent-pool-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("Reversi/checkpoints/latest.pt"),
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("Reversi/checkpoints/submission.py"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainerConfig(
        games_per_iteration=args.games,
        minibatch_size=args.batch_size,
        dvn_epochs=args.dvn_epochs,
        dpn_epochs=args.dpn_epochs,
        dpn_learning_rate=args.dpn_learning_rate,
        entropy_weight=args.entropy_weight,
        ppo_clip_range=args.ppo_clip,
        ppo_target_kl=args.ppo_target_kl,
        rollout_temperature=args.temperature,
        dvn_replay_capacity=args.dvn_replay_capacity,
        dvn_replay_fraction=args.dvn_replay_fraction,
        opponent_pool_size=args.opponent_pool_size,
        opponent_pool_fraction=args.opponent_pool_fraction,
        device=args.device,
        seed=args.seed,
    )
    trainer = SelfPlayTrainer(DVN(), DPN(), config)
    trainer.add_policy_snapshot()
    print(
        json.dumps(
            {
                "device": str(trainer.device),
                "dvn_parameters": trainer.dvn.parameter_count(),
                "dpn_parameters": trainer.dpn.parameter_count(),
            }
        )
    )
    for iteration in range(1, args.iterations + 1):
        metrics = trainer.train_iteration()
        print(json.dumps({"iteration": iteration, **metrics}, sort_keys=True))
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trainer.checkpoint(iteration), args.checkpoint)
        export_dpn_submission(trainer.dpn, args.submission)


if __name__ == "__main__":
    main()
