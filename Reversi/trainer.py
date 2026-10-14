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
from .model import DPN, DVN, dvn_loss, final_board_classes, ppo_loss, temporal_deltas
from .symmetry import transform_action


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    games_per_iteration: int = 1024
    minibatch_size: int = 1024
    dvn_epochs: int = 1
    dpn_epochs: int = 4
    dvn_learning_rate: float = 3e-4
    dpn_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gamma: float = 1.0
    entropy_weight: float = 0.01
    ppo_clip_range: float = 0.2
    ppo_target_kl: float = 0.02
    rollout_temperature: float = 1.0
    dvn_replay_capacity: int = 262_144
    dvn_replay_fraction: float = 0.5
    opponent_pool_size: int = 4
    opponent_pool_fraction: float = 0.5
    use_amp: bool = True
    device: str = "auto"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.rollout_temperature <= 0:
            raise ValueError("rollout_temperature must be positive for PPO")
        if not 0 < self.ppo_clip_range < 1:
            raise ValueError("ppo_clip_range must be in (0, 1)")
        if self.dvn_replay_capacity < 0:
            raise ValueError("dvn_replay_capacity must be non-negative")
        if not 0 <= self.dvn_replay_fraction < 1:
            raise ValueError("dvn_replay_fraction must be in [0, 1)")
        if self.opponent_pool_size < 0:
            raise ValueError("opponent_pool_size must be non-negative")
        if not 0 <= self.opponent_pool_fraction <= 1:
            raise ValueError("opponent_pool_fraction must be in [0, 1]")


@dataclass(slots=True)
class RolloutBatch:
    boards: NDArray[np.int8]
    players: NDArray[np.int8]
    legal_masks: NDArray[np.bool_]
    policy_boards: NDArray[np.int8]
    policy_legal_masks: NDArray[np.bool_]
    actions: NDArray[np.int64]
    old_log_probabilities: NDArray[np.float32]
    policy_trainable: NDArray[np.bool_]
    next_boards: NDArray[np.int8]
    next_players: NDArray[np.int8]
    next_legal_masks: NDArray[np.bool_]
    terminated: NDArray[np.bool_]
    rewards: NDArray[np.float32]
    final_boards: NDArray[np.int8]
    outcomes: NDArray[np.float32]
    winners: NDArray[np.int8]
    opponent_pool_games: int

    def __len__(self) -> int:
        return self.boards.shape[0]


@dataclass(slots=True)
class TensorRollout:
    boards: Tensor
    players: Tensor
    legal_masks: Tensor
    policy_boards: Tensor
    policy_legal_masks: Tensor
    actions: Tensor
    old_log_probabilities: Tensor
    policy_trainable: Tensor
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
        policy_boards=move(rollout.policy_boards),
        policy_legal_masks=move(rollout.policy_legal_masks),
        actions=move(rollout.actions),
        old_log_probabilities=move(rollout.old_log_probabilities),
        policy_trainable=move(rollout.policy_trainable),
        next_boards=move(rollout.next_boards),
        next_players=move(rollout.next_players),
        next_legal_masks=move(rollout.next_legal_masks),
        terminated=move(rollout.terminated),
        rewards=move(rollout.rewards),
        final_boards=move(rollout.final_boards),
        outcomes=move(rollout.outcomes),
    )


@dataclass(slots=True)
class DVNReplayBatch:
    boards: NDArray[np.int8]
    players: NDArray[np.int8]
    legal_masks: NDArray[np.bool_]
    final_boards: NDArray[np.int8]
    outcomes: NDArray[np.float32]

    def __len__(self) -> int:
        return self.boards.shape[0]


@dataclass(slots=True)
class TensorDVNBatch:
    boards: Tensor
    players: Tensor
    legal_masks: Tensor
    final_boards: Tensor
    outcomes: Tensor

    def __len__(self) -> int:
        return self.boards.shape[0]


class DVNReplayBuffer:
    """Fixed-size uniform reservoir of completed self-play positions."""

    _FIELDS = ("boards", "players", "legal_masks", "final_boards", "outcomes")

    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.priorities = np.empty(0, dtype=np.float64)
        self.boards = np.empty((0, BOARD_CELLS), dtype=np.int8)
        self.players = np.empty(0, dtype=np.int8)
        self.legal_masks = np.empty((0, BOARD_CELLS + 1), dtype=np.bool_)
        self.final_boards = np.empty((0, BOARD_CELLS), dtype=np.int8)
        self.outcomes = np.empty(0, dtype=np.float32)

    def __len__(self) -> int:
        return self.boards.shape[0]

    def add(self, rollout: RolloutBatch) -> None:
        if self.capacity == 0 or len(rollout) == 0:
            return
        priorities = self.rng.random(len(rollout))
        joined_priorities = np.concatenate((self.priorities, priorities))
        joined = {
            field: np.concatenate((getattr(self, field), getattr(rollout, field)))
            for field in self._FIELDS
        }
        if joined_priorities.size > self.capacity:
            keep = np.argpartition(joined_priorities, -self.capacity)[-self.capacity :]
            joined_priorities = joined_priorities[keep]
            joined = {field: values[keep] for field, values in joined.items()}
        self.priorities = joined_priorities
        for field, values in joined.items():
            setattr(self, field, values)

    def sample(self, count: int) -> DVNReplayBatch | None:
        count = min(count, len(self))
        if count <= 0:
            return None
        indices = self.rng.choice(len(self), size=count, replace=False)
        return DVNReplayBatch(
            **{field: getattr(self, field)[indices] for field in self._FIELDS}
        )

    def state_dict(self) -> dict[str, object]:
        state: dict[str, object] = {
            "capacity": self.capacity,
            "priorities": torch.from_numpy(self.priorities),
        }
        state.update(
            {field: torch.from_numpy(getattr(self, field)) for field in self._FIELDS}
        )
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        # The current run's capacity is authoritative, so a large old buffer can
        # be resumed with a smaller memory budget.
        priorities = state["priorities"].cpu().numpy()
        keep = np.arange(priorities.shape[0])
        if keep.size > self.capacity:
            keep = np.argpartition(priorities, -self.capacity)[-self.capacity :]
        self.priorities = priorities[keep].copy()
        for field in self._FIELDS:
            values = state[field].cpu().numpy()
            setattr(self, field, values[keep].copy())


def _to_tensor_dvn_batch(
    batch: DVNReplayBatch, device: torch.device
) -> TensorDVNBatch:
    def move(array: NDArray) -> Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).to(device, non_blocking=True)

    return TensorDVNBatch(
        boards=move(batch.boards),
        players=move(batch.players),
        legal_masks=move(batch.legal_masks),
        final_boards=move(batch.final_boards),
        outcomes=move(batch.outcomes),
    )


def _build_symmetry_maps() -> tuple[Tensor, Tensor, Tensor]:
    old_to_new = np.empty((8, BOARD_CELLS + 1), dtype=np.int64)
    source_for_destination = np.empty((8, BOARD_CELLS), dtype=np.int64)
    for symmetry in range(8):
        for old_action in range(BOARD_CELLS):
            new_action = transform_action(old_action, symmetry)
            old_to_new[symmetry, old_action] = new_action
            source_for_destination[symmetry, new_action] = old_action
        old_to_new[symmetry, PASS_ACTION] = PASS_ACTION
    action_source = np.empty((8, BOARD_CELLS + 1), dtype=np.int64)
    action_source[:, :BOARD_CELLS] = source_for_destination
    action_source[:, PASS_ACTION] = PASS_ACTION
    return (
        torch.from_numpy(old_to_new),
        torch.from_numpy(source_for_destination),
        torch.from_numpy(action_source),
    )


_ACTION_TRANSFORMS, _GRID_SOURCES, _ACTION_SOURCES = _build_symmetry_maps()


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
        self.replay_buffer = DVNReplayBuffer(
            self.config.dvn_replay_capacity, seed=self.config.seed + 17
        )
        self.policy_pool: list[DPN] = []

    def add_policy_snapshot(self) -> None:
        """Freeze the current DPN as a possible future self-play opponent."""
        if self.config.opponent_pool_size == 0:
            return
        snapshot = copy.deepcopy(self.dpn).to(self.device).eval()
        for parameter in snapshot.parameters():
            parameter.requires_grad_(False)
        self.policy_pool.append(snapshot)
        if len(self.policy_pool) > self.config.opponent_pool_size:
            del self.policy_pool[: -self.config.opponent_pool_size]

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
        use_pool = bool(self.policy_pool) and self.config.opponent_pool_fraction > 0
        pool_games = np.zeros(env.num_envs, dtype=np.bool_)
        opponent_colors = np.zeros(env.num_envs, dtype=np.int8)
        opponent_policy: DPN | None = None
        if use_pool:
            pool_games = np.random.random(env.num_envs) < self.config.opponent_pool_fraction
            opponent_colors[pool_games] = np.random.choice(
                np.array([-1, 1], dtype=np.int8), size=int(pool_games.sum())
            )
            opponent_policy = self.policy_pool[np.random.randint(len(self.policy_pool))]
        records: dict[str, list[NDArray]] = {
            key: []
            for key in (
                "boards", "players", "legal_masks", "policy_boards",
                "policy_legal_masks", "actions", "old_log_probabilities",
                "policy_trainable",
                "next_boards", "next_players", "next_legal_masks",
                "terminated", "rewards", "game_ids",
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
            count = boards.shape[0]
            symmetries = torch.randint(0, 8, (count,), device=self.device)
            sources = _GRID_SOURCES.to(self.device)[symmetries]
            policy_boards = board_tensor.gather(1, sources)
            policy_cells = mask_tensor[:, :BOARD_CELLS].gather(1, sources)
            policy_masks = torch.cat(
                (policy_cells, mask_tensor[:, PASS_ACTION:]), dim=1
            )
            opponent_turns = pool_games[game_ids] & (
                players == opponent_colors[game_ids]
            )
            trainable = ~opponent_turns
            trainable_tensor = torch.from_numpy(trainable).to(self.device)
            opponent_tensor = ~trainable_tensor
            logits = torch.empty(
                (count, BOARD_CELLS + 1), device=self.device, dtype=torch.float32
            )
            with self._autocast():
                if trainable_tensor.any():
                    logits[trainable_tensor] = self.dpn(
                        policy_boards[trainable_tensor],
                        player_tensor[trainable_tensor],
                        policy_masks[trainable_tensor],
                    ).policy_logits.float()
                if opponent_tensor.any():
                    if opponent_policy is None:
                        raise RuntimeError("opponent policy was not selected")
                    logits[opponent_tensor] = opponent_policy(
                        policy_boards[opponent_tensor],
                        player_tensor[opponent_tensor],
                        policy_masks[opponent_tensor],
                    ).policy_logits.float()
            temperature = self.config.rollout_temperature
            behavior_log_probabilities = torch.log_softmax(
                logits.float() / temperature, dim=1
            )
            selected = torch.multinomial(
                behavior_log_probabilities.exp(), 1
            ).squeeze(1)
            old_log_probabilities = behavior_log_probabilities.gather(
                1, selected[:, None]
            ).squeeze(1)
            env_actions = _ACTION_SOURCES.to(self.device)[symmetries, selected]
            actions = selected.cpu().numpy().astype(np.int64, copy=False)
            original_actions = env_actions.cpu().numpy().astype(np.int64, copy=False)

            all_actions = np.full(env.num_envs, PASS_ACTION, dtype=np.int64)
            all_actions[game_ids] = original_actions
            step_result = env.step(all_actions)
            next_observation = env.observe(copy=False)

            records["boards"].append(boards)
            records["players"].append(players)
            records["legal_masks"].append(masks)
            records["policy_boards"].append(
                policy_boards.cpu().numpy().astype(np.int8, copy=False)
            )
            records["policy_legal_masks"].append(
                policy_masks.cpu().numpy().astype(np.bool_, copy=False)
            )
            records["actions"].append(actions)
            records["old_log_probabilities"].append(
                old_log_probabilities.cpu().numpy().astype(np.float32, copy=False)
            )
            records["policy_trainable"].append(trainable)
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
            policy_boards=joined["policy_boards"],
            policy_legal_masks=joined["policy_legal_masks"],
            actions=joined["actions"],
            old_log_probabilities=joined["old_log_probabilities"],
            policy_trainable=joined["policy_trainable"],
            next_boards=joined["next_boards"],
            next_players=joined["next_players"],
            next_legal_masks=joined["next_legal_masks"],
            terminated=joined["terminated"],
            rewards=joined["rewards"],
            final_boards=final_boards_by_game[sample_game_ids],
            outcomes=outcomes,
            winners=winners,
            opponent_pool_games=int(pool_games.sum()),
        )

    def _minibatches(self, count: int) -> Iterator[Tensor]:
        permutation = torch.randperm(count, device=self.device)
        size = self.config.minibatch_size
        for start in range(0, count, size):
            yield permutation[start : start + size]

    def train_dvn(
        self, rollout: TensorRollout | TensorDVNBatch
    ) -> dict[str, float]:
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
        # Dropout would make the likelihood ratio noisy even before an update.
        # eval() disables it while still allowing gradients through all weights.
        self.dpn.eval()
        eligible = torch.nonzero(
            rollout.policy_trainable, as_tuple=False
        ).flatten()
        if eligible.numel() == 0:
            raise RuntimeError("rollout contains no actions from the trainable policy")
        train_advantages = advantages[eligible]
        train_advantages = (train_advantages - train_advantages.mean()) / (
            train_advantages.std(unbiased=False) + 1e-6
        )
        totals = {
            "dpn_policy": 0.0,
            "dpn_entropy": 0.0,
            "dpn_approx_kl": 0.0,
            "dpn_clip_fraction": 0.0,
            "dpn_ratio_mean": 0.0,
            "dpn_total": 0.0,
        }
        batches = 0
        epochs_completed = 0
        for _ in range(self.config.dpn_epochs):
            epoch_kl = 0.0
            epoch_batches = 0
            for relative_indices in self._minibatches(eligible.numel()):
                indices = eligible[relative_indices]
                self.dpn_optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    output = self.dpn(
                        rollout.policy_boards[indices],
                        rollout.players[indices],
                        rollout.policy_legal_masks[indices],
                    )
                    losses = ppo_loss(
                        output,
                        rollout.actions[indices],
                        rollout.old_log_probabilities[indices],
                        train_advantages[relative_indices],
                        clip_range=self.config.ppo_clip_range,
                        entropy_weight=self.config.entropy_weight,
                        temperature=self.config.rollout_temperature,
                    )
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.dpn.parameters(), 1.0)
                self.dpn_optimizer.step()
                for name in (
                    "policy", "entropy", "approx_kl", "clip_fraction",
                    "ratio_mean", "total",
                ):
                    totals[f"dpn_{name}"] += float(losses[name].detach())
                epoch_kl += float(losses["approx_kl"].detach())
                epoch_batches += 1
                batches += 1
            epochs_completed += 1
            mean_epoch_kl = epoch_kl / epoch_batches
            if (
                self.config.ppo_target_kl > 0
                and mean_epoch_kl > 1.5 * self.config.ppo_target_kl
            ):
                break
        metrics = {key: value / batches for key, value in totals.items()}
        metrics["dpn_epochs_completed"] = float(epochs_completed)
        metrics["dpn_train_positions"] = int(eligible.numel())
        return metrics

    def _make_dvn_training_batch(
        self, rollout: TensorRollout, replay: DVNReplayBatch | None
    ) -> TensorDVNBatch:
        if replay is None:
            return TensorDVNBatch(
                boards=rollout.boards,
                players=rollout.players,
                legal_masks=rollout.legal_masks,
                final_boards=rollout.final_boards,
                outcomes=rollout.outcomes,
            )
        replay_tensor = _to_tensor_dvn_batch(replay, self.device)
        return TensorDVNBatch(
            boards=torch.cat((rollout.boards, replay_tensor.boards)),
            players=torch.cat((rollout.players, replay_tensor.players)),
            legal_masks=torch.cat((rollout.legal_masks, replay_tensor.legal_masks)),
            final_boards=torch.cat((rollout.final_boards, replay_tensor.final_boards)),
            outcomes=torch.cat((rollout.outcomes, replay_tensor.outcomes)),
        )

    def train_iteration(self) -> dict[str, float | int | str]:
        rollout_cpu = self.collect_self_play()
        rollout = _to_tensor_rollout(rollout_cpu, self.device)
        replay_count = 0
        replay_fraction = self.config.dvn_replay_fraction
        if replay_fraction > 0:
            replay_count = round(len(rollout) * replay_fraction / (1 - replay_fraction))
        replay = self.replay_buffer.sample(replay_count)
        dvn_batch = self._make_dvn_training_batch(rollout, replay)
        metrics: dict[str, float | int | str] = {
            "device": str(self.device),
            "games": self.config.games_per_iteration,
            "positions": len(rollout),
            "black_wins": int((rollout_cpu.winners == 1).sum()),
            "white_wins": int((rollout_cpu.winners == -1).sum()),
            "draws": int((rollout_cpu.winners == 0).sum()),
            "opponent_pool_games": rollout_cpu.opponent_pool_games,
            "policy_pool_size": len(self.policy_pool),
            "dvn_replay_positions": 0 if replay is None else len(replay),
            "dvn_train_positions": len(dvn_batch),
        }
        metrics.update(self.train_dvn(dvn_batch))
        self.replay_buffer.add(rollout_cpu)
        metrics["dvn_replay_size"] = len(self.replay_buffer)

        # DPN sees a fixed critic for the entire policy update.
        target_dvn = copy.deepcopy(self.dvn).to(self.device).eval()
        for parameter in target_dvn.parameters():
            parameter.requires_grad_(False)
        advantages = self.compute_advantages(target_dvn, rollout)
        metrics["advantage_mean"] = float(advantages.mean())
        metrics["advantage_std"] = float(advantages.std(unbiased=False))
        metrics.update(self.train_dpn(rollout, advantages))
        del target_dvn, dvn_batch, rollout
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
            "model_config": asdict(self.dpn.config),
            "dvn": self.dvn.state_dict(),
            "dpn": self.dpn.state_dict(),
            "dvn_optimizer": self.dvn_optimizer.state_dict(),
            "dpn_optimizer": self.dpn_optimizer.state_dict(),
            "dvn_replay": self.replay_buffer.state_dict(),
            "policy_pool": [policy.state_dict() for policy in self.policy_pool],
        }

    def load_checkpoint(self, checkpoint: dict[str, object]) -> int:
        """Restore trainable state while retaining this run's configuration."""
        self.dvn.load_state_dict(checkpoint["dvn"])
        self.dpn.load_state_dict(checkpoint["dpn"])
        if "dvn_optimizer" in checkpoint:
            self.dvn_optimizer.load_state_dict(checkpoint["dvn_optimizer"])
        if "dpn_optimizer" in checkpoint:
            self.dpn_optimizer.load_state_dict(checkpoint["dpn_optimizer"])
        if "dvn_replay" in checkpoint and self.config.dvn_replay_capacity > 0:
            self.replay_buffer.load_state_dict(checkpoint["dvn_replay"])
        self.policy_pool.clear()
        for state in checkpoint.get("policy_pool", []):
            policy = DPN(self.dpn.config).to(self.device).eval()
            policy.load_state_dict(state)
            for parameter in policy.parameters():
                parameter.requires_grad_(False)
            self.policy_pool.append(policy)
        if self.config.opponent_pool_size == 0:
            self.policy_pool.clear()
        elif len(self.policy_pool) > self.config.opponent_pool_size:
            del self.policy_pool[: -self.config.opponent_pool_size]
        return int(checkpoint.get("iteration", 0))
