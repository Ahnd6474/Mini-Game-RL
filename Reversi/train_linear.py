"""Bootstrap the tree-search leaf evaluator from parallel random games."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .batch_env import BatchReversiEnv
from .env import BOARD_CELLS, PASS_ACTION
from .linear_value import LinearWinPredictor, linear_value_loss, save_linear_evaluator
from .symmetry import transform_action
from .trainer import resolve_device


@dataclass(slots=True)
class LinearDataset:
    """Compact state samples labeled from the player-to-move perspective."""

    boards: NDArray[np.int8]
    outcomes: NDArray[np.float32]
    game_ids: NDArray[np.int32]
    winners: NDArray[np.int8]

    def __len__(self) -> int:
        return self.boards.shape[0]


def collect_random_games(num_games: int, seed: int = 0) -> LinearDataset:
    """Play all games in parallel and retain every pre-action position."""
    if num_games < 2:
        raise ValueError("num_games must be at least 2")
    rng = np.random.default_rng(seed)
    env = BatchReversiEnv(num_games)
    board_chunks: list[NDArray[np.int8]] = []
    player_chunks: list[NDArray[np.int8]] = []
    game_chunks: list[NDArray[np.int32]] = []

    for _ in range(128):
        if not env.active.any():
            break
        observation = env.observe(copy=False)
        active_ids = np.flatnonzero(env.active)
        board_chunks.append(observation.boards[active_ids].copy())
        player_chunks.append(observation.current_players[active_ids].copy())
        game_chunks.append(active_ids.astype(np.int32, copy=False))

        # Taking the argmax of iid random scores samples legal moves uniformly.
        masks = observation.legal_action_masks
        scores = rng.random(masks.shape, dtype=np.float32)
        scores[~masks] = -1.0
        actions = scores.argmax(axis=1).astype(np.int64, copy=False)
        actions[~env.active] = PASS_ACTION
        env.step(actions)
    else:
        raise RuntimeError("random self-play exceeded the Reversi action bound")

    boards = np.concatenate(board_chunks)
    players = np.concatenate(player_chunks)
    game_ids = np.concatenate(game_chunks)
    winners = env.winners()
    # A single color-relative representation makes one set of weights valid for
    # both black and white. Labels are also relative to the player about to move.
    boards = (boards * players[:, None]).astype(np.int8, copy=False)
    outcomes = (winners[game_ids] * players).astype(np.float32, copy=False)
    return LinearDataset(boards, outcomes, game_ids, winners)


def _symmetry_sources() -> Tensor:
    sources = np.empty((8, BOARD_CELLS), dtype=np.int64)
    for symmetry in range(8):
        for source in range(BOARD_CELLS):
            destination = transform_action(source, symmetry)
            sources[symmetry, destination] = source
    return torch.from_numpy(sources)


_GRID_SOURCES = _symmetry_sources()


@torch.inference_mode()
def evaluate(
    model: LinearWinPredictor,
    boards: Tensor,
    outcomes: Tensor,
    indices: Tensor,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    decisive_correct = 0
    decisive_count = 0
    for start in range(0, indices.numel(), batch_size):
        selected = indices[start : start + batch_size]
        logits = model(boards[selected])
        target = outcomes[selected]
        loss_sum += float(linear_value_loss(logits, target)) * selected.numel()
        decisive = target != 0
        decisive_correct += int(((logits > 0) == (target > 0))[decisive].sum())
        decisive_count += int(decisive.sum())
    return {
        "loss": loss_sum / max(1, indices.numel()),
        "decisive_accuracy": decisive_correct / max(1, decisive_count),
    }


def train_linear_model(
    dataset: LinearDataset,
    *,
    device: torch.device,
    epochs: int = 12,
    batch_size: int = 8192,
    learning_rate: float = 3e-2,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[LinearWinPredictor, dict[str, object]]:
    """Fit the convex 64-weight model, splitting complete games by ID."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")

    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    game_order = torch.randperm(dataset.winners.size, generator=generator).numpy()
    validation_games = max(1, int(dataset.winners.size * validation_fraction))
    is_validation_game = np.zeros(dataset.winners.size, dtype=np.bool_)
    is_validation_game[game_order[:validation_games]] = True
    validation_mask = is_validation_game[dataset.game_ids]
    train_indices = torch.from_numpy(np.flatnonzero(~validation_mask)).to(device)
    validation_indices = torch.from_numpy(np.flatnonzero(validation_mask)).to(device)

    # int8 keeps a million-state bootstrap dataset small on the GPU; each batch
    # is converted to the model dtype only inside LinearWinPredictor.forward.
    boards = torch.from_numpy(np.ascontiguousarray(dataset.boards)).to(device)
    outcomes = torch.from_numpy(np.ascontiguousarray(dataset.outcomes)).to(device)
    symmetry_sources = _GRID_SOURCES.to(device)
    model = LinearWinPredictor().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = train_indices[torch.randperm(train_indices.numel(), device=device)]
        loss_sum = 0.0
        for start in range(0, order.numel(), batch_size):
            selected = order[start : start + batch_size]
            batch = boards[selected]
            symmetries = torch.randint(0, 8, (selected.numel(),), device=device)
            batch = batch.gather(1, symmetry_sources[symmetries])
            logits = model(batch)
            loss = linear_value_loss(logits, outcomes[selected])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * selected.numel()

        # The true game has exact D4 symmetry. Projection removes finite-sample
        # noise and leaves only ten independent positional weight orbits.
        model.symmetrize_()
        validation = evaluate(
            model, boards, outcomes, validation_indices, batch_size
        )
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": loss_sum / train_indices.numel(),
            "validation_loss": validation["loss"],
            "validation_decisive_accuracy": validation["decisive_accuracy"],
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, sort_keys=True), flush=True)

    metrics: dict[str, object] = {
        "games": int(dataset.winners.size),
        "positions": len(dataset),
        "train_positions": int(train_indices.numel()),
        "validation_positions": int(validation_indices.numel()),
        "draw_rate": float(np.mean(dataset.winners == 0)),
        "seconds": time.perf_counter() - started,
        "device": str(device),
        "history": history,
    }
    return model, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Reversi/checkpoints/linear_value.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collection_started = time.perf_counter()
    dataset = collect_random_games(args.games, args.seed)
    collection_seconds = time.perf_counter() - collection_started
    print(
        json.dumps(
            {
                "games": args.games,
                "positions": len(dataset),
                "collection_seconds": collection_seconds,
                "positions_per_second": len(dataset) / collection_seconds,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    device = resolve_device(args.device)
    model, metrics = train_linear_model(
        dataset,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    metrics["collection_seconds"] = collection_seconds
    output = save_linear_evaluator(model, args.output, metrics)
    weights = np.asarray(model.weights()).reshape(8, 8)
    print(json.dumps({"saved": str(output), "weights": weights.tolist()}))


if __name__ == "__main__":
    main()
