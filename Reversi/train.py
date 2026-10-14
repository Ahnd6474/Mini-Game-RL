"""Command-line entry point for parallel DVN/DPN self-play training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import DPN, DVN
from .trainer import SelfPlayTrainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--dvn-epochs", type=int, default=1)
    parser.add_argument("--dpn-epochs", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("Reversi/checkpoints/latest.pt"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainerConfig(
        games_per_iteration=args.games,
        minibatch_size=args.batch_size,
        dvn_epochs=args.dvn_epochs,
        dpn_epochs=args.dpn_epochs,
        rollout_temperature=args.temperature,
        device=args.device,
        seed=args.seed,
    )
    trainer = SelfPlayTrainer(DVN(), DPN(), config)
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


if __name__ == "__main__":
    main()
