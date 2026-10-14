"""Benchmark end-to-end self-play training configurations on the local GPU."""

from __future__ import annotations

import argparse
import copy
import json
import time

import torch

from .model import DPN, DVN
from .trainer import SelfPlayTrainer, TrainerConfig, _to_tensor_rollout


def _timed(device: torch.device, operation):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def benchmark(games: int, batch_size: int, seed: int) -> dict[str, float | int | str]:
    config = TrainerConfig(
        games_per_iteration=games,
        minibatch_size=batch_size,
        dvn_epochs=1,
        dpn_epochs=1,
        device="cuda",
        seed=seed,
    )
    trainer = SelfPlayTrainer(DVN(), DPN(), config)
    torch.cuda.reset_peak_memory_stats(trainer.device)

    rollout_cpu, rollout_seconds = _timed(trainer.device, trainer.collect_self_play)
    rollout, upload_seconds = _timed(
        trainer.device,
        lambda: _to_tensor_rollout(rollout_cpu, trainer.device),
    )
    dvn_metrics, dvn_seconds = _timed(
        trainer.device,
        lambda: trainer.train_dvn(rollout),
    )

    target_dvn = copy.deepcopy(trainer.dvn).to(trainer.device).eval()
    for parameter in target_dvn.parameters():
        parameter.requires_grad_(False)
    advantages, delta_seconds = _timed(
        trainer.device,
        lambda: trainer.compute_advantages(target_dvn, rollout),
    )
    dpn_metrics, dpn_seconds = _timed(
        trainer.device,
        lambda: trainer.train_dpn(rollout, advantages),
    )
    total_seconds = (
        rollout_seconds + upload_seconds + dvn_seconds + delta_seconds + dpn_seconds
    )
    positions = len(rollout)
    return {
        "games": games,
        "batch_size": batch_size,
        "positions": positions,
        "rollout_s": round(rollout_seconds, 4),
        "upload_s": round(upload_seconds, 4),
        "dvn_s": round(dvn_seconds, 4),
        "delta_s": round(delta_seconds, 4),
        "dpn_s": round(dpn_seconds, 4),
        "total_s": round(total_seconds, 4),
        "positions_per_second": round(positions / total_seconds),
        "train_samples_per_second": round(2 * positions / (dvn_seconds + dpn_seconds)),
        "cuda_peak_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2),
        **{key: round(value, 5) for key, value in dvn_metrics.items()},
        **{key: round(value, 5) for key, value in dpn_metrics.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        default="1024:512,1024:1024,2048:2048,4096:2048,4096:4096",
        help="comma-separated games:minibatch pairs",
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    for index, item in enumerate(args.configs.split(",")):
        games, batch_size = (int(value) for value in item.split(":"))
        try:
            result = benchmark(games, batch_size, args.seed + index)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            result = {
                "games": games,
                "batch_size": batch_size,
                "error": "CUDA out of memory",
            }
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
