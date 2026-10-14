"""Fixed validation and head-to-head evaluation for DVN/DPN training."""

from __future__ import annotations

import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional as F

from .batch_env import BatchReversiEnv
from .env import BLACK, BOARD_CELLS, PASS_ACTION, WHITE, ReversiEnv
from .model import DPN, DVN, final_board_classes
from .tree_agent import LinearEvaluator, TreeSearchAgent


@dataclass(slots=True)
class DVNValidationSet:
    boards: NDArray[np.int8]
    players: NDArray[np.int8]
    legal_masks: NDArray[np.bool_]
    final_boards: NDArray[np.int8]
    outcomes: NDArray[np.float32]

    def __len__(self) -> int:
        return self.boards.shape[0]


def collect_dvn_validation(
    num_games: int = 512,
    max_positions: int = 8192,
    seed: int = 12345,
) -> DVNValidationSet:
    """Create a fixed random-policy validation set, labeled after each game."""
    if num_games < 1 or max_positions < 1:
        raise ValueError("num_games and max_positions must be positive")
    rng = np.random.default_rng(seed)
    env = BatchReversiEnv(num_games)
    board_chunks: list[NDArray[np.int8]] = []
    player_chunks: list[NDArray[np.int8]] = []
    mask_chunks: list[NDArray[np.bool_]] = []
    game_chunks: list[NDArray[np.int32]] = []

    for _ in range(128):
        if not env.active.any():
            break
        observation = env.observe(copy=False)
        active_ids = np.flatnonzero(env.active)
        board_chunks.append(observation.boards[active_ids].copy())
        player_chunks.append(observation.current_players[active_ids].copy())
        mask_chunks.append(observation.legal_action_masks[active_ids].copy())
        game_chunks.append(active_ids.astype(np.int32, copy=False))

        scores = rng.random(observation.legal_action_masks.shape, dtype=np.float32)
        scores[~observation.legal_action_masks] = -1.0
        actions = scores.argmax(axis=1).astype(np.int64, copy=False)
        actions[~env.active] = PASS_ACTION
        env.step(actions)
    else:
        raise RuntimeError("validation self-play exceeded the action bound")

    boards = np.concatenate(board_chunks)
    players = np.concatenate(player_chunks)
    legal_masks = np.concatenate(mask_chunks)
    game_ids = np.concatenate(game_chunks)
    final_by_game = env.boards.reshape(num_games, BOARD_CELLS)
    winners = env.winners()
    final_boards = final_by_game[game_ids]
    outcomes = (winners[game_ids] * players).astype(np.float32, copy=False)

    if boards.shape[0] > max_positions:
        chosen = np.sort(
            rng.choice(boards.shape[0], size=max_positions, replace=False)
        )
        boards = boards[chosen]
        players = players[chosen]
        legal_masks = legal_masks[chosen]
        final_boards = final_boards[chosen]
        outcomes = outcomes[chosen]
    return DVNValidationSet(
        boards, players, legal_masks, final_boards.copy(), outcomes
    )


def _autocast(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def evaluate_dvn(
    model: DVN,
    dataset: DVNValidationSet,
    device: torch.device,
    batch_size: int = 2048,
    use_amp: bool = True,
) -> dict[str, float]:
    """Evaluate final-cell classification and scalar outcome prediction."""
    model.eval()
    board_loss_sum = 0.0
    value_squared_sum = 0.0
    value_absolute_sum = 0.0
    cell_correct = 0
    decisive_correct = 0
    decisive_count = 0
    sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0.0

    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        boards = torch.from_numpy(dataset.boards[start:stop]).to(device)
        players = torch.from_numpy(dataset.players[start:stop]).to(device)
        masks = torch.from_numpy(dataset.legal_masks[start:stop]).to(device)
        finals = torch.from_numpy(dataset.final_boards[start:stop]).to(device)
        outcomes = torch.from_numpy(dataset.outcomes[start:stop]).to(device)
        targets = final_board_classes(finals, players)
        with _autocast(device, use_amp):
            output = model(boards, players, masks)
        logits = output.final_board_logits.float()
        values = output.value.float()
        board_loss_sum += float(
            F.cross_entropy(
                logits.reshape(-1, 3), targets.reshape(-1), reduction="sum"
            )
        )
        errors = values - outcomes
        value_squared_sum += float(errors.square().sum())
        value_absolute_sum += float(errors.abs().sum())
        cell_correct += int((logits.argmax(dim=-1) == targets).sum())
        decisive = outcomes != 0
        decisive_correct += int(
            ((values > 0) == (outcomes > 0))[decisive].sum()
        )
        decisive_count += int(decisive.sum())
        sum_x += float(values.sum())
        sum_y += float(outcomes.sum())
        sum_x2 += float(values.square().sum())
        sum_y2 += float(outcomes.square().sum())
        sum_xy += float((values * outcomes).sum())

    count = len(dataset)
    board_loss = board_loss_sum / (count * BOARD_CELLS)
    value_mse = value_squared_sum / count
    covariance = sum_xy - sum_x * sum_y / count
    variance_x = max(0.0, sum_x2 - sum_x * sum_x / count)
    variance_y = max(0.0, sum_y2 - sum_y * sum_y / count)
    correlation = covariance / max(1e-12, (variance_x * variance_y) ** 0.5)
    return {
        "dvn_validation_board": board_loss,
        "dvn_validation_value": value_mse,
        "dvn_validation_total": board_loss + 0.5 * value_mse,
        "dvn_validation_cell_accuracy": cell_correct / (count * BOARD_CELLS),
        "dvn_validation_value_mae": value_absolute_sum / count,
        "dvn_validation_decisive_accuracy": decisive_correct
        / max(1, decisive_count),
        "dvn_validation_value_correlation": correlation,
    }


def make_fixed_openings(
    count: int = 4,
    plies: int = 8,
    seed: int = 24680,
) -> list[tuple[tuple[int, ...], int, int]]:
    """Build deterministic varied openings; each is later played with both colors."""
    if count < 1 or plies < 0:
        raise ValueError("count must be positive and plies nonnegative")
    rng = random.Random(seed)
    openings: list[tuple[tuple[int, ...], int, int]] = []
    while len(openings) < count:
        env = ReversiEnv()
        for _ in range(plies):
            if env.terminated:
                break
            env.step(rng.choice(env.observe().legal_actions))
        if not env.terminated:
            openings.append((env.board, env.current_player, env.turn))
    return openings


@torch.inference_mode()
def evaluate_dpn_vs_tree(
    model: DPN,
    evaluator: LinearEvaluator,
    *,
    device: torch.device,
    depth: int = 4,
    openings: list[tuple[tuple[int, ...], int, int]] | None = None,
    use_amp: bool = True,
) -> dict[str, float | int]:
    """Play paired-color deterministic matches against the fixed tree anchor."""
    openings = openings or make_fixed_openings()
    model.eval()
    wins = draws = losses = 0
    margin_sum = 0.0
    tree_nodes = 0
    tree_seconds = 0.0
    dpn_seconds = 0.0

    for board, current_player, turn in openings:
        for dpn_color in (BLACK, WHITE):
            env = ReversiEnv(board, current_player=current_player, turn=turn)
            tree = TreeSearchAgent(evaluator, depth=depth)
            while not env.terminated:
                observation = env.observe()
                if observation.current_player == dpn_color:
                    started = time.perf_counter()
                    boards = torch.tensor([observation.board], device=device)
                    players = torch.tensor(
                        [observation.current_player], device=device
                    )
                    masks = torch.tensor(
                        [observation.legal_action_mask], device=device
                    ).bool()
                    with _autocast(device, use_amp):
                        action = int(
                            model(boards, players, masks).policy_logits.argmax(dim=1)
                        )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    dpn_seconds += time.perf_counter() - started
                else:
                    started = time.perf_counter()
                    action = tree.select_action(
                        observation.board, observation.current_player
                    )
                    tree_seconds += time.perf_counter() - started
                    tree_nodes += tree.stats.nodes
                env.step(action)

            black_count, white_count = env.disc_counts()
            margin = (black_count - white_count) * dpn_color
            margin_sum += margin
            if margin > 0:
                wins += 1
            elif margin < 0:
                losses += 1
            else:
                draws += 1

    games = wins + draws + losses
    return {
        "tree_games": games,
        "tree_wins": wins,
        "tree_draws": draws,
        "tree_losses": losses,
        "tree_score": (wins + 0.5 * draws) / games,
        "tree_mean_disc_margin": margin_sum / games,
        "tree_nodes": tree_nodes,
        "tree_nodes_per_second": tree_nodes / max(tree_seconds, 1e-12),
        "tree_search_seconds": tree_seconds,
        "dpn_inference_seconds": dpn_seconds,
    }


def load_tree_evaluator(path: Path | None) -> LinearEvaluator:
    return LinearEvaluator.from_json(path) if path else LinearEvaluator()
