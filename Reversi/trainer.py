"""GPU-oriented parallel self-play and separate DVN/DPN optimization."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Iterator

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .batch_env import BatchReversiEnv
from .env import BOARD_CELLS, PASS_ACTION
from .model import DPN, DVN, dpn_loss, dvn_loss, final_board_classes, temporal_deltas
from .symmetry import transform_action


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    games_per_iteration: int = 1024
    minibatch_size: int = 1024
    dvn_epochs: int = 3
    dpn_epochs: int = 3
    dvn_learning_rate: float = 3e-4
    dpn_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gamma: float = 1.0
    entropy_weight: float = 0.01
    rollout_temperature: float = 1.0
    use_amp: bool = True
    device: str = "auto"
    seed: int = 0


@dataclass(slots=True)
class RolloutBatch:
    boards: NDArray[np.int8]
    players: NDArray[np.int8]
    legal_masks: NDArray[np.bool_]
    actions: NDArray[np.int64]
    next_boards: NDArray[np.int8]
    next_players: NDArray[np.int8]
    next_legal_masks: NDArray[np.bool_]
    terminated: NDArray[np.bool_]
    rewards: NDArray[np.float32]
    final_boards: NDArray[np.int8]
    outcomes: NDArray[np.float32]
    winners: NDArray[np.int8]

    def __len__(self) -> int:
        return self.boards.shape[0]


@dataclass(slots=True)
class TensorRollout:
    boards: Tensor
    players: Tensor
    legal_masks: Tensor
    actions: Tensor
    next_boards: Tensor
    next_players: Tensor
    next_legal_masks: Tensor
    terminated: Tensor
    rewards: Tensor
    final_boards: Tensor
    outcomes: Tensor

    def __len__(self) -> int:
        return self.boards.shape[0]


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this PyTorch build has no CUDA support. "
            "Install a CUDA wheel from download.pytorch.org."
        )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    return device


def _to_tensor_rollout(rollout: RolloutBatch, device: torch.device) -> TensorRollout:
    def move(array: NDArray) -> Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).to(device, non_blocking=True)

    return TensorRollout(
        boards=move(rollout.boards),
        players=move(rollout.players),
        legal_masks=move(rollout.legal_masks),
        actions=move(rollout.actions),
        next_boards=move(rollout.next_boards),
        next_players=move(rollout.next_players),
        next_legal_masks=move(rollout.next_legal_masks),
        terminated=move(rollout.terminated),
        rewards=move(rollout.rewards),
        final_boards=move(rollout.final_boards),
        outcomes=move(rollout.outcomes),
    )


def _build_symmetry_maps() -> tuple[Tensor, Tensor]:
    old_to_new = np.empty((8, BOARD_CELLS + 1), dtype=np.int64)
    source_for_destination = np.empty((8, BOARD_CELLS), dtype=np.int64)
    for symmetry in range(8):
        for old_action in range(BOARD_CELLS):
            new_action = transform_action(old_action, symmetry)
            old_to_new[symmetry, old_action] = new_action
            source_for_destination[symmetry, new_action] = old_action
        old_to_new[symmetry, PASS_ACTION] = PASS_ACTION
    return torch.from_numpy(old_to_new), torch.from_numpy(source_for_destination)


_ACTION_TRANSFORMS, _GRID_SOURCES = _build_symmetry_maps()


def _augment_tensors(
    boards: Tensor,
    legal_masks: Tensor,
    *,
    final_boards: Tensor | None = None,
    actions: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """Apply one independently sampled D4 symmetry per sample on-device."""
    count = boards.shape[0]
    symmetries = torch.randint(0, 8, (count,), device=boards.device)
    sources = _GRID_SOURCES.to(boards.device)[symmetries]
    boards = boards.gather(1, sources)
    legal_cells = legal_masks[:, :BOARD_CELLS].gather(1, sources)
    legal_masks = torch.cat((legal_cells, legal_masks[:, PASS_ACTION:]), dim=1)
    if final_boards is not None:
        final_boards = final_boards.gather(1, sources)
    if actions is not None:
        actions = _ACTION_TRANSFORMS.to(boards.device)[symmetries, actions]
    return boards, legal_masks, final_boards, actions


class SelfPlayTrainer:
    """Alternates supervised DVN learning and frozen-DVN DPN learning."""

    def __init__(
        self,
        dvn: DVN,
        dpn: DPN,
        config: TrainerConfig | None = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.device = resolve_device(self.config.device)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)
        self.dvn = dvn.to(self.device)
        self.dpn = dpn.to(self.device)
        self.dvn_optimizer = torch.optim.AdamW(
            self.dvn.parameters(),
            lr=self.config.dvn_learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.dpn_optimizer = torch.optim.AdamW(
            self.dpn.parameters(),
            lr=self.config.dpn_learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _autocast(self):
        if self.device.type != "cuda" or not self.config.use_amp:
            return nullcontext()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    @torch.inference_mode()
    def collect_self_play(self) -> RolloutBatch:
        """Run all games together: NumPy rules on CPU, DPN batches on GPU."""
        env = BatchReversiEnv(self.config.games_per_iteration)
        self.dpn.eval()
        records: dict[str, list[NDArray]] = {
            key: []
            for key in (
                "boards", "players", "legal_masks", "actions", "next_boards",
                "next_players", "next_legal_masks", "terminated", "rewards",
                "game_ids",
            )
        }
        step_count = 0
        while env.active.any():
            if step_count >= 128:
                raise RuntimeError("self-play exceeded the Reversi action bound")
            observation = env.observe(copy=False)
            game_ids = np.flatnonzero(env.active)
            boards = observation.boards[game_ids].copy()
            players = observation.current_players[game_ids].copy()
            masks = observation.legal_action_masks[game_ids].copy()

            board_tensor = torch.from_numpy(boards).to(self.device)
            player_tensor = torch.from_numpy(players).to(self.device)
            mask_tensor = torch.from_numpy(masks).to(self.device)
            with self._autocast():
                logits = self.dpn(board_tensor, player_tensor, mask_tensor).policy_logits
            temperature = self.config.rollout_temperature
            if temperature <= 0:
                selected = logits.argmax(dim=1)
            else:
                selected = torch.multinomial(
                    torch.softmax(logits.float() / temperature, dim=1), 1
                ).squeeze(1)
            actions = selected.cpu().numpy().astype(np.int64, copy=False)

            all_actions = np.full(env.num_envs, PASS_ACTION, dtype=np.int64)
            all_actions[game_ids] = actions
            step_result = env.step(all_actions)
            next_observation = env.observe(copy=False)

            records["boards"].append(boards)
            records["players"].append(players)
            records["legal_masks"].append(masks)
            records["actions"].append(actions)
            records["next_boards"].append(next_observation.boards[game_ids].copy())
            records["next_players"].append(
                next_observation.current_players[game_ids].copy()
            )
            records["next_legal_masks"].append(
                next_observation.legal_action_masks[game_ids].copy()
            )
            records["terminated"].append(step_result.newly_terminated[game_ids].copy())
            records["rewards"].append(
                step_result.actor_rewards[game_ids].astype(np.float32)
            )
            records["game_ids"].append(game_ids.astype(np.int64))
            step_count += 1

        joined = {key: np.concatenate(value) for key, value in records.items()}
        final_boards_by_game = env.boards.reshape(env.num_envs, BOARD_CELLS).copy()
        winners = env.winners()
        sample_game_ids = joined["game_ids"]
        outcomes = (
            winners[sample_game_ids] * joined["players"]
        ).astype(np.float32)
        return RolloutBatch(
            boards=joined["boards"],
            players=joined["players"],
            legal_masks=joined["legal_masks"],
            actions=joined["actions"],
            next_boards=joined["next_boards"],
            next_players=joined["next_players"],
            next_legal_masks=joined["next_legal_masks"],
            terminated=joined["terminated"],
            rewards=joined["rewards"],
            final_boards=final_boards_by_game[sample_game_ids],
            outcomes=outcomes,
            winners=winners,
        )

    def _minibatches(self, count: int) -> Iterator[Tensor]:
        permutation = torch.randperm(count, device=self.device)
        size = self.config.minibatch_size
        for start in range(0, count, size):
            yield permutation[start : start + size]

    def train_dvn(self, rollout: TensorRollout) -> dict[str, float]:
        self.dvn.train()
        totals = {"dvn_board": 0.0, "dvn_value": 0.0, "dvn_total": 0.0}
        batches = 0
        for _ in range(self.config.dvn_epochs):
            for indices in self._minibatches(len(rollout)):
                boards, masks, final_boards, _ = _augment_tensors(
                    rollout.boards[indices],
                    rollout.legal_masks[indices],
                    final_boards=rollout.final_boards[indices],
                )
                players = rollout.players[indices]
                targets = final_board_classes(final_boards, players)
                self.dvn_optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    output = self.dvn(boards, players, masks)
                    losses = dvn_loss(output, targets, rollout.outcomes[indices])
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.dvn.parameters(), 1.0)
                self.dvn_optimizer.step()
                for name in ("board", "value", "total"):
                    totals[f"dvn_{name}"] += float(losses[name].detach())
                batches += 1
        return {key: value / batches for key, value in totals.items()}

    @torch.inference_mode()
    def compute_advantages(self, target_dvn: DVN, rollout: TensorRollout) -> Tensor:
        target_dvn.eval()
        values = torch.empty(len(rollout), device=self.device)
        next_values = torch.empty_like(values)
        size = self.config.minibatch_size
        for start in range(0, len(rollout), size):
            stop = min(start + size, len(rollout))
            selection = slice(start, stop)
            with self._autocast():
                values[selection] = target_dvn(
                    rollout.boards[selection],
                    rollout.players[selection],
                    rollout.legal_masks[selection],
                ).value.float()
                next_values[selection] = target_dvn(
                    rollout.next_boards[selection],
                    rollout.next_players[selection],
                    rollout.next_legal_masks[selection],
                ).value.float()
        return temporal_deltas(
            values,
            next_values,
            rollout.rewards,
            rollout.terminated,
            self.config.gamma,
        )

    def train_dpn(
        self, rollout: TensorRollout, advantages: Tensor
    ) -> dict[str, float]:
        self.dpn.train()
        totals = {"dpn_policy": 0.0, "dpn_entropy": 0.0, "dpn_total": 0.0}
        batches = 0
        for _ in range(self.config.dpn_epochs):
            for indices in self._minibatches(len(rollout)):
                boards, masks, _, actions = _augment_tensors(
                    rollout.boards[indices],
                    rollout.legal_masks[indices],
                    actions=rollout.actions[indices],
                )
                self.dpn_optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    output = self.dpn(boards, rollout.players[indices], masks)
                    losses = dpn_loss(
                        output,
                        actions,
                        advantages[indices],
                        entropy_weight=self.config.entropy_weight,
                    )
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.dpn.parameters(), 1.0)
                self.dpn_optimizer.step()
                for name in ("policy", "entropy", "total"):
                    totals[f"dpn_{name}"] += float(losses[name].detach())
                batches += 1
        return {key: value / batches for key, value in totals.items()}

    def train_iteration(self) -> dict[str, float | int | str]:
        rollout_cpu = self.collect_self_play()
        rollout = _to_tensor_rollout(rollout_cpu, self.device)
        metrics: dict[str, float | int | str] = {
            "device": str(self.device),
            "games": self.config.games_per_iteration,
            "positions": len(rollout),
            "black_wins": int((rollout_cpu.winners == 1).sum()),
            "white_wins": int((rollout_cpu.winners == -1).sum()),
            "draws": int((rollout_cpu.winners == 0).sum()),
        }
        metrics.update(self.train_dvn(rollout))

        # DPN sees a fixed critic for the entire policy update.
        target_dvn = copy.deepcopy(self.dvn).to(self.device).eval()
        for parameter in target_dvn.parameters():
            parameter.requires_grad_(False)
        advantages = self.compute_advantages(target_dvn, rollout)
        metrics["advantage_mean"] = float(advantages.mean())
        metrics["advantage_std"] = float(advantages.std(unbiased=False))
        metrics.update(self.train_dpn(rollout, advantages))
        del target_dvn, rollout
        if self.device.type == "cuda":
            metrics["cuda_peak_memory_mb"] = int(
                torch.cuda.max_memory_allocated(self.device) / 1024**2
            )
            torch.cuda.reset_peak_memory_stats(self.device)
        return metrics

    def checkpoint(self, iteration: int) -> dict[str, object]:
        return {
            "iteration": iteration,
            "trainer_config": asdict(self.config),
            "dvn": self.dvn.state_dict(),
            "dpn": self.dpn.state_dict(),
            "dvn_optimizer": self.dvn_optimizer.state_dict(),
            "dpn_optimizer": self.dpn_optimizer.state_dict(),
        }
