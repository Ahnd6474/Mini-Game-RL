"""Run a measured DVN/DPN experiment and produce checkpoints and curves."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .evaluation import (
    collect_dvn_validation,
    evaluate_dpn_vs_tree,
    evaluate_dvn,
    load_tree_evaluator,
    make_fixed_openings,
)
from .export_submission import export_dpn_submission
from .model import DPN, DVN, ReversiTransformerConfig
from .trainer import SelfPlayTrainer, TrainerConfig
from .training_plot import write_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--dvn-epochs", type=int, default=1)
    parser.add_argument("--dpn-epochs", type=int, default=4)
    parser.add_argument("--dpn-learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--ppo-target-kl", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dvn-replay-capacity", type=int, default=262_144)
    parser.add_argument("--dvn-replay-fraction", type=float, default=0.5)
    parser.add_argument("--opponent-pool-size", type=int, default=4)
    parser.add_argument("--opponent-pool-fraction", type=float, default=0.5)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-openings", type=int, default=4)
    parser.add_argument("--tree-depth", type=int, default=4)
    parser.add_argument("--validation-games", type=int, default=512)
    parser.add_argument("--validation-positions", type=int, default=8192)
    parser.add_argument(
        "--tree-weights",
        type=Path,
        default=Path("Reversi/checkpoints/linear_value.json"),
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("Reversi/runs/latest")
    )
    return parser.parse_args()


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.iterations < 1 or args.eval_every < 1:
        raise ValueError("iterations and eval-every must be positive")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.run_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    # Models are constructed before SelfPlayTrainer receives them, so seed here
    # as well as inside the trainer to make initialization reproducible.
    checkpoint = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
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
    model_config = (
        ReversiTransformerConfig(**checkpoint["model_config"])
        if checkpoint is not None
        else ReversiTransformerConfig()
    )
    trainer = SelfPlayTrainer(DVN(model_config), DPN(model_config), config)
    start_iteration = 0
    if checkpoint is not None:
        start_iteration = trainer.load_checkpoint(checkpoint)
    if not trainer.policy_pool:
        trainer.add_policy_snapshot()
    validation = collect_dvn_validation(
        args.validation_games, args.validation_positions, seed=args.seed + 1000
    )
    openings = make_fixed_openings(
        args.eval_openings, plies=8, seed=args.seed + 2000
    )
    tree_evaluator = load_tree_evaluator(args.tree_weights)
    metadata = {
        "trainer_config": asdict(config),
        "iterations": args.iterations,
        "start_iteration": start_iteration,
        "resume": None if args.resume is None else str(args.resume),
        "eval_every": args.eval_every,
        "tree_depth": args.tree_depth,
        "tree_weights": str(args.tree_weights),
        "validation_positions": len(validation),
        "evaluation_games": 2 * len(openings),
        "dvn_parameters": trainer.dvn.parameter_count(),
        "dpn_parameters": trainer.dpn.parameter_count(),
    }
    (args.run_dir / "config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    initial: dict[str, object] = {
        "iteration": start_iteration,
        "elapsed_seconds": 0.0,
        "policy_pool_size": len(trainer.policy_pool),
        "dvn_replay_size": len(trainer.replay_buffer),
    }
    initial.update(
        evaluate_dvn(
            trainer.dvn,
            validation,
            trainer.device,
            batch_size=args.batch_size,
            use_amp=config.use_amp,
        )
    )
    initial.update(
        evaluate_dpn_vs_tree(
            trainer.dpn,
            tree_evaluator,
            device=trainer.device,
            depth=args.tree_depth,
            openings=openings,
            use_amp=config.use_amp,
        )
    )
    records: list[dict[str, object]] = [initial]
    _append_jsonl(metrics_path, initial)
    print(json.dumps(initial, sort_keys=True), flush=True)

    experiment_started = time.perf_counter()
    best_score = float(initial["tree_score"])
    baseline_checkpoint = trainer.checkpoint(start_iteration)
    baseline_checkpoint["metrics"] = initial
    torch.save(baseline_checkpoint, args.run_dir / "best_tree.pt")
    export_dpn_submission(trainer.dpn, args.run_dir / "submission_best.py")
    final_iteration = start_iteration + args.iterations
    for iteration in range(start_iteration + 1, final_iteration + 1):
        iteration_started = time.perf_counter()
        record: dict[str, object] = {"iteration": iteration}
        record.update(trainer.train_iteration())
        iteration_seconds = time.perf_counter() - iteration_started
        record["iteration_seconds"] = iteration_seconds
        record["positions_per_second"] = (
            int(record["positions"]) / iteration_seconds
        )
        record["elapsed_seconds"] = time.perf_counter() - experiment_started
        record.update(
            evaluate_dvn(
                trainer.dvn,
                validation,
                trainer.device,
                batch_size=args.batch_size,
                use_amp=config.use_amp,
            )
        )
        should_match = (
            (iteration - start_iteration) % args.eval_every == 0
            or iteration == final_iteration
        )
        if should_match:
            record.update(
                evaluate_dpn_vs_tree(
                    trainer.dpn,
                    tree_evaluator,
                    device=trainer.device,
                    depth=args.tree_depth,
                    openings=openings,
                    use_amp=config.use_amp,
                )
            )

        improved = (
            "tree_score" in record and float(record["tree_score"]) > best_score
        )
        if improved:
            best_score = float(record["tree_score"])
            trainer.add_policy_snapshot()
        saved_checkpoint = trainer.checkpoint(iteration)
        saved_checkpoint["metrics"] = record
        torch.save(saved_checkpoint, args.run_dir / "latest.pt")
        if improved:
            torch.save(saved_checkpoint, args.run_dir / "best_tree.pt")
            export_dpn_submission(
                trainer.dpn, args.run_dir / "submission_best.py"
            )
        export_dpn_submission(trainer.dpn, args.run_dir / "submission.py")
        records.append(record)
        _append_jsonl(metrics_path, record)
        write_training_curves(records, args.run_dir / "learning_curves.svg")
        print(json.dumps(record, sort_keys=True), flush=True)

    summary = {
        "iterations": args.iterations,
        "start_iteration": start_iteration,
        "final_iteration": final_iteration,
        "games_trained": args.iterations * args.games,
        "positions_trained": sum(int(record.get("positions", 0)) for record in records),
        "total_seconds": time.perf_counter() - experiment_started,
        "initial": records[0],
        "final": records[-1],
        "best_tree_score": max(
            float(record["tree_score"])
            for record in records
            if "tree_score" in record
        ),
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
