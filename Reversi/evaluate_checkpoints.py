"""Evaluate one or more saved DPN checkpoints against fixed tree agents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .evaluation import evaluate_dpn_vs_tree, load_tree_evaluator, make_fixed_openings
from .model import DPN, ReversiTransformerConfig
from .trainer import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--depths", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--openings", type=int, default=16)
    parser.add_argument("--opening-plies", type=int, default=8)
    parser.add_argument("--seed", type=int, default=31014)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--tree-weights",
        type=Path,
        default=Path("Reversi/checkpoints/linear_value.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = resolve_device(args.device)
    evaluator = load_tree_evaluator(args.tree_weights)
    openings = make_fixed_openings(
        args.openings, plies=args.opening_plies, seed=args.seed
    )
    results: list[dict[str, object]] = []
    for checkpoint_path in args.checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = ReversiTransformerConfig(**checkpoint["model_config"])
        model = DPN(config).to(device)
        model.load_state_dict(checkpoint["dpn"])
        for depth in args.depths:
            metrics = evaluate_dpn_vs_tree(
                model,
                evaluator,
                device=device,
                depth=depth,
                openings=openings,
            )
            record: dict[str, object] = {
                "checkpoint": str(checkpoint_path),
                "iteration": int(checkpoint["iteration"]),
                "tree_depth": depth,
                **metrics,
            }
            results.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
