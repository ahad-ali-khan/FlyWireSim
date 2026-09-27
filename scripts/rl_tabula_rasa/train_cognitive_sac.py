"""Recurrent SAC trainer for the Phase 9 cognitive policy.

Replaces the PPO update in train_cognitive.py with Soft Actor-Critic:
  - Replay segments include the complete episode prefix for recurrent context.
  - Independent recurrent twin critics and slowly updated target encoders.
  - Automatic entropy tuning and finite, checkpointed experience replay.

CLI flags and output format match train_cognitive.py exactly.
Checkpoint files are NOT interchangeable with PPO checkpoints.
"""

from __future__ import annotations

import argparse
import math
import json
import os
import random
import signal
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_

from .baselines import random_action
from .cognitive import (BELIEF_LOSS_WEIGHT, COGNITIVE_ARCH_ENABLED,
                        SELF_MODEL_LOSS_WEIGHT, CognitiveActorCritic,
                        SPECIALIST_NAMES, added_parameters,
                        workspace_dominants_by_role)
from .environment_config import OBSTACLE_SEED
from .env import PHYSICS_VERSION, REFERENCE_DT
from .novelty import NoveltyEnv
from .training_runtime import DATA, TRAINING_MAX_THREADS, atomic_json

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CHECKPOINT = DATA / "cognitive_sac_checkpoint.pt"

# SAC hyperparameters
SAC_LEARNING_RATE = 3e-4
SAC_GAMMA = 0.99
SAC_TAU = 0.005
SAC_BATCH_EPISODES = 8
SAC_SEGMENT_LEN = 16
SAC_UPDATES_PER_EP = 16
SAC_WARMUP_EPISODES = 20
REPLAY_CAPACITY = 20_000
CHECKPOINT_INTERVAL_EPISODES = 20
SNAPSHOT_INTERVAL_EPISODES = 100
LIVE_FRAME_STREAM = DATA / "live_state_stream.jsonl"
# At alpha=.1, ~2 nats/tick over 70 ticks can outweigh the +1 catch.
# Start below the terminal reward scale; automatic tuning remains enabled.
LOG_ALPHA_INIT = math.log(0.005)
GRAD_CLIP = 0.5
TRAINER_VERSION = "recurrent_sac_v2"

STOP_REQUESTED = False


class TwinCritic(nn.Module):
    """Independent observation-history encoders, trained by Bellman error.

    Do not feed detached, constantly changing actor embeddings to the critics.
    Each target critic owns a slowly updated copy of its encoder as well.
    """

    def __init__(self, observation_dim: int = 19, action_dim: int = 3):
        super().__init__()
        self.encoders = nn.ModuleList([nn.GRU(observation_dim, 64, batch_first=True)
                                       for _ in range(2)])

        def head():
            return nn.Sequential(
                nn.Linear(64 + action_dim, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1),
            )

        self.q1 = head()
        self.q2 = head()

    def encode(self, observations):
        return tuple(encoder(observations)[0] for encoder in self.encoders)

    def forward(self, history, action):
        return tuple(head(torch.cat((encoded, action), dim=-1)).squeeze(-1)
                     for head, encoded in zip((self.q1, self.q2), history))

    def q_min(self, history, action):
        q1, q2 = self.forward(history, action)
        return torch.minimum(q1, q2)


class EpisodeReplayBuffer:
    """Store full episodes and sample contiguous segments for recurrent learning."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.episodes: deque[dict] = deque()
        self._total = 0

    def push(self, episode: dict):
        length = len(episode["obs"])
        while self._total + length > self.capacity and self.episodes:
            self._total -= len(self.episodes.popleft()["obs"])
        self.episodes.append(episode)
        self._total += length

    def state_dict(self):
        return {"capacity": self.capacity, "episodes": list(self.episodes),
                "total": self._total}

    def load_state_dict(self, state):
        capacity = int(state.get("capacity", self.capacity))
        if capacity < 1:
            raise ValueError("replay buffer capacity must be positive")
        episodes = deque({key: np.asarray(value).copy() for key, value in episode.items()}
                         for episode in state["episodes"])
        total = sum(len(episode["obs"]) for episode in episodes)
        while total > capacity and episodes:
            total -= len(episodes.popleft()["obs"])
        self.capacity = capacity
        self.episodes = episodes
        self._total = total

    @property
    def episode_count(self):
        return len(self.episodes)

    def sample_segments(self, n_episodes: int, segment_len: int,
                        rng: np.random.Generator):
        n_episodes = min(n_episodes, len(self.episodes))
        chosen = rng.choice(len(self.episodes), size=n_episodes, replace=False)
        segments = {key: [] for key in self.episodes[0]}
        for index in chosen:
            episode = self.episodes[index]
            length = min(segment_len, len(episode["obs"]))
            start = rng.integers(0, len(episode["obs"]) - length + 1)
            # Include the prefix from the true reset, but apply losses only to
            # the chosen window. Never start a mid-episode GRU at zero.
            for key, values in episode.items():
                segments[key].append(np.asarray(values[:start + length]))
            segments.setdefault("loss_mask", []).append(
                np.arange(start + length) >= start)
        padded = {}
        valid = []
        for key, arrays in segments.items():
            max_len = max(array.shape[0] for array in arrays)
            padded_arrays = []
            for array in arrays:
                padding = max_len - array.shape[0]
                if key == "obs":
                    valid.append(np.concatenate((np.ones(array.shape[0], dtype=bool),
                                                 np.zeros(padding, dtype=bool))))
                if padding:
                    array = np.concatenate((array, np.zeros(
                        (padding, *array.shape[1:]), dtype=array.dtype)))
                padded_arrays.append(array)
            padded[key] = np.stack(padded_arrays)
        padded["valid"] = np.stack(valid) & padded.pop("loss_mask")
        return padded

    def __len__(self):
        return self._total


def sample_squashed(mean, log_std):
    """Numerically stable tanh-Gaussian sample and change-of-variables log p."""
    distribution = Normal(mean, log_std.clamp(-5, 2).exp().expand_as(mean))
    raw = distribution.rsample()
    logp = (distribution.log_prob(raw) -
            2 * (math.log(2) - raw - F.softplus(-2 * raw))).sum(-1)
    return raw.tanh(), logp


def legacy_moth_observation(observation):
    """The frozen PPO moth was trained with velocities in metres per tick."""
    observation = observation.copy()
    observation[3:6] *= REFERENCE_DT
    observation[14:17] *= REFERENCE_DT
    return observation


def bat_policy_observation(observation, frame="body"):
    """Express measured sonar vectors in the bat's own forward/left/up frame.

    A coordinate transform, not a steering controller: no target information
    beyond the masked observation, no teacher actions, no policy bypass.
    """
    if frame == "world":
        return observation.copy()
    observation = observation.copy()
    forward = observation[6:9]
    left = np.array([-forward[1], forward[0], 0], dtype=np.float32)
    left /= max(float(np.linalg.norm(left)), 1e-8)
    basis = np.stack((forward, left, np.cross(forward, left)))
    observation[11:14] = basis @ observation[11:14]
    observation[14:17] = basis @ observation[14:17]
    return observation


def _sac_update(actor, critic, critic_target, log_alpha, actor_opt, critic_opt,
                alpha_opt, buffer, target_entropy, rng):
    from torch.distributions import Normal

    data = buffer.sample_segments(SAC_BATCH_EPISODES, SAC_SEGMENT_LEN, rng)
    tensor = lambda key, dtype=torch.float32: torch.as_tensor(data[key], dtype=dtype)
    obs = tensor("obs")
    actions = tensor("actions")
    rewards = tensor("rewards")
    dones = tensor("dones")
    next_obs = tensor("next_obs")
    next_own = tensor("next_own")
    opponent = tensor("opponent")
    visible = tensor("visible", torch.bool)
    valid = tensor("valid", torch.bool)
    valid_flat = valid.reshape(-1)
    valid_count = valid_flat.sum().clamp_min(1)
    alpha = log_alpha.exp().detach()

    actor.train()
    # One recurrent unroll gives correctly aligned h_t AND h_(t+1).
    history_obs = torch.cat((obs[:, :1], next_obs), dim=1)
    output = actor.forward_sequence(history_obs)
    new_actions, new_logp = sample_squashed(output.mean[:, :-1], actor.log_standard_deviation)

    with torch.no_grad():
        next_actions, next_logp = sample_squashed(output.mean[:, 1:], actor.log_standard_deviation)
        target_history = tuple(h[:, 1:] for h in critic_target.encode(history_obs))
        target_q = critic_target.q_min(target_history, next_actions)
        target = rewards + SAC_GAMMA * (1.0 - dones) * (target_q - alpha * next_logp)

    history = critic.encode(obs)
    q1, q2 = critic(history, actions)
    squared_error = ((q1 - target).square() + (q2 - target).square()).reshape(-1)
    critic_loss = (squared_error * valid_flat).sum() / valid_count
    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    clip_grad_norm_(critic.parameters(), GRAD_CLIP)
    critic_opt.step()

    flat_logp = new_logp.reshape(-1)
    with torch.no_grad():
        history = critic.encode(obs)
    critic.requires_grad_(False)
    q_new = critic.q_min(history, new_actions).reshape(-1)
    actor_loss = ((alpha * flat_logp - q_new) * valid_flat).sum() / valid_count
    self_error = (output.predicted_next_state[:, :-1] - next_own).square().mean(dim=-1).reshape(-1)
    self_loss = (self_error * valid_flat).sum() / valid_count
    belief_error = (output.belief_state[:, :-1] - opponent).square().mean(dim=-1)
    belief_mask = visible & valid
    belief_loss = belief_error[belief_mask].mean() if belief_mask.any() else belief_error.new_zeros(())
    total_actor_loss = (actor_loss + SELF_MODEL_LOSS_WEIGHT * self_loss
                        + BELIEF_LOSS_WEIGHT * belief_loss)
    actor_opt.zero_grad(set_to_none=True)
    total_actor_loss.backward()
    clip_grad_norm_(actor.parameters(), GRAD_CLIP)
    actor_opt.step()
    critic.requires_grad_(True)
    with torch.no_grad():
        actor.log_standard_deviation.clamp_(-5, 2)

    alpha_loss = -(log_alpha * (flat_logp.detach() + target_entropy) * valid_flat).sum() / valid_count
    alpha_opt.zero_grad(set_to_none=True)
    alpha_loss.backward()
    alpha_opt.step()

    with torch.no_grad():
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.mul_(1.0 - SAC_TAU).add_(SAC_TAU * param)

    actor.eval()
    return {
        "critic_loss": float(critic_loss.detach()),
        "actor_loss": float(actor_loss.detach()),
        "alpha": float(log_alpha.exp().detach()),
        "alpha_loss": float(alpha_loss.detach()),
        "self_model_mse": float(self_loss.detach()),
        "belief_mse": float(belief_loss.detach()),
        "entropy": float(-(flat_logp * valid_flat).sum().detach() / valid_count),
    }


def _atomic_torch_save(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temp = Path(handle.name)
    try:
        torch.save(payload, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _opponent_state(env, role):
    other = "moth" if role == "bat" else "bat"
    body = getattr(env, other)
    return np.concatenate((body.position, body.velocity)).astype(np.float32)


def _analysis(episodes, key, names_by_role):
    result = {}
    for role in ("bat", "moth"):
        selected = [episode for episode in episodes if episode.get(key)]
        if not selected:
            result[role] = {}
            continue
        width = len(names_by_role[role])
        phases = {"early": [], "mid": [], "final_20_ticks": []}
        for episode in selected:
            values = np.zeros((episode["hunt_length"], width), dtype=np.float64)
            source = episode[key][role]
            values[:min(len(source), len(values))] = source[:len(values)]
            length = len(values)
            phases["early"].append(values[:max(1, length // 3)].mean(axis=0))
            phases["mid"].append(values[length // 3:max(length // 3 + 1,
                                                        2 * length // 3)].mean(axis=0))
            phases["final_20_ticks"].append(values[max(0, length - 20):].mean(axis=0))
        result[role] = {phase: {name: float(np.mean([row[i] for row in rows]))
                                for i, name in enumerate(names_by_role[role])}
                        for phase, rows in phases.items()}
    return result


def _write_outputs(episodes, models, seed, finished, output_path, moth_opponent_checkpoint,
                   environment_config, moth_mode, initial_alpha, observation_frame):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_agent_params = {role: added_parameters(role, model.observation_dim)
                        for role, model in models.items()}
    opponent_mix = {}
    for mode in ("learned", "random", "stationary"):
        selected = [ep for ep in episodes if ep.get("moth_opponent_mode", "learned") == mode]
        opponent_mix[mode] = {
            "episodes": len(selected),
            "fraction": len(selected) / len(episodes) if episodes else None,
            "catch_rate": float(np.mean([ep["caught"] for ep in selected])) if selected else None,
        }
    payload = {
        "phase": 9, "architecture": "cognitive recurrent SAC",
        "trainer_version": TRAINER_VERSION, "physics_version": PHYSICS_VERSION,
        "environment_config": environment_config, "requested_moth_mode": moth_mode,
        "novelty_reward_enabled": False,
        "initial_alpha": initial_alpha,
        "observation_frame": observation_frame,
        "cognitive_arch_enabled": COGNITIVE_ARCH_ENABLED,
        "seed": seed, "completed_episodes": len(episodes),
        "training_roles": ["bat"],
        "frozen_moth_opponent_checkpoint": moth_opponent_checkpoint,
        "moth_opponent_mix": opponent_mix,
        "sac": {"learning_rate": SAC_LEARNING_RATE, "gamma": SAC_GAMMA,
                "tau": SAC_TAU, "batch_episodes": SAC_BATCH_EPISODES,
                "segment_len": SAC_SEGMENT_LEN, "updates_per_episode": SAC_UPDATES_PER_EP,
                "warmup_episodes": SAC_WARMUP_EPISODES,
                "replay_capacity": REPLAY_CAPACITY},
        "parameters": per_agent_params,
        "total_added_parameters": sum(v["added_parameters"] for v in per_agent_params.values()),
        "episodes": episodes,
        "status": "complete" if finished else "stopped",
    }
    atomic_json(output_path, payload)
    attention_names = {
        "bat": ["position_x", "position_y", "position_z", "velocity_x", "velocity_y", "velocity_z",
                "heading_x", "heading_y", "heading_z", "sonar_hit", "moth_distance", "moth_direction_x",
                "moth_direction_y", "moth_direction_z", "relative_velocity_x", "relative_velocity_y",
                "relative_velocity_z", "time_remaining", "sonar_occluded"],
        "moth": ["position_x", "position_y", "position_z", "velocity_x", "velocity_y", "velocity_z",
                 "heading_x", "heading_y", "heading_z", "bat_visible", "bat_direction_x", "bat_direction_y",
                 "bat_direction_z", "bat_distance", "relative_velocity_x", "relative_velocity_y",
                 "relative_velocity_z", "flame_direction_x", "flame_direction_y", "flame_direction_z",
                 "flame_distance", "time_remaining", "vision_occluded"]}
    workspace_names = {role: list(SPECIALIST_NAMES) for role in ("bat", "moth")}
    atomic_json(output_path.with_name(f"{output_path.stem}_attention_analysis.json"),
                {"phase": 9, "attention_by_hunt_phase": _analysis(
                    episodes, "attention_weights_by_role", attention_names)})
    atomic_json(output_path.with_name(f"{output_path.stem}_workspace_analysis.json"),
                {"phase": 9, "workspace_dominance_by_hunt_phase": _analysis(
                    episodes, "workspace_weights_by_role", workspace_names)})
    belief = {}
    for role in ("bat", "moth"):
        errors, occluded_count = [], 0
        for episode in episodes:
            estimates = np.asarray(episode["belief_estimates_by_role"][role], dtype=np.float32)
            actual = np.asarray(episode["opponent_states_by_role"][role], dtype=np.float32)
            occluded = np.asarray(episode["occlusion_by_role"][role], dtype=bool)
            occluded_count += int(occluded.sum())
            if len(actual) == len(estimates) and occluded.any():
                errors.extend(np.linalg.norm(estimates[occluded] - actual[occluded], axis=-1).tolist())
        visible_losses = [ep["belief_mse_when_visible"][role] for ep in episodes
                          if ep.get("belief_mse_when_visible", {}).get(role) is not None]
        belief[role] = {"ticks_occluded": occluded_count,
            "mean_drift_distance_during_occlusion": float(np.mean(errors)) if errors else None,
            "mean_supervised_belief_mse": float(np.mean(visible_losses)) if visible_losses else None}
    atomic_json(output_path.with_name(f"{output_path.stem}_belief_analysis.json"),
                {"phase": 9, "belief_tracking": belief,
                 "note": "Belief MSE is supervised only on visible ticks; drift analysis uses logged estimates on occluded ticks."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--moth-opponent-checkpoint", type=Path, required=True,
                        help="frozen moth checkpoint; bat trains against 50/30/20 mix")
    default_output = RESULTS / f"cognitive_sac_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--no-live", action="store_true", help="do not replace the Blender playback stream")
    parser.add_argument("--moth-mode", choices=("mix", "stationary", "random"), default="mix")
    parser.add_argument("--initial-alpha", type=float, default=0.005)
    parser.add_argument("--observation-frame", choices=("body", "world"), default="body")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if not 0 < args.initial_alpha <= 1:
        parser.error("--initial-alpha must be in (0, 1]")
    args.checkpoint = args.checkpoint or DATA / f"{args.output.stem}.pt"
    if not args.resume and (args.checkpoint.exists() or args.output.exists()):
        parser.error("fresh runs require unused checkpoint/output paths; refusing to overwrite results")
    if args.resume and not args.checkpoint.exists():
        parser.error("resume checkpoint does not exist")

    torch.manual_seed(args.seed)
    torch.set_num_threads(TRAINING_MAX_THREADS)
    np.random.seed(args.seed)
    random.seed(args.seed)
    sample_rng = np.random.default_rng(args.seed + 7)
    moth_action_rng = np.random.default_rng(args.seed + 901)
    env = NoveltyEnv("bat", config={"natural_environment": True,
        "obstacle_seed": args.seed if OBSTACLE_SEED is None else OBSTACLE_SEED})

    bat = CognitiveActorCritic("bat", 19).eval()
    moth = CognitiveActorCritic("moth", 23).eval()
    opponent_path = args.moth_opponent_checkpoint.resolve()
    if not opponent_path.is_file():
        parser.error(f"moth checkpoint not found: {opponent_path}")
    checkpoint_data = torch.load(opponent_path, map_location="cpu", weights_only=False)
    if "moth" not in checkpoint_data.get("models", {}):
        parser.error(f"checkpoint has no moth policy: {opponent_path}")
    moth.load_state_dict(checkpoint_data["models"]["moth"])
    models = {"bat": bat, "moth": moth}

    critic = TwinCritic()
    critic_target = TwinCritic()
    critic_target.load_state_dict(critic.state_dict())
    for param in critic_target.parameters():
        param.requires_grad_(False)
    log_alpha = torch.tensor(math.log(args.initial_alpha), requires_grad=True)
    target_entropy = -3.0
    actor_opt = torch.optim.Adam(bat.parameters(), lr=SAC_LEARNING_RATE)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=SAC_LEARNING_RATE)
    alpha_opt = torch.optim.Adam([log_alpha], lr=SAC_LEARNING_RATE)
    buffer = EpisodeReplayBuffer(REPLAY_CAPACITY)
    episodes = []
    start = 0
    completed_updates = 0

    if args.resume and args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if (state.get("trainer_version") != TRAINER_VERSION or
                state.get("physics_version") != PHYSICS_VERSION or
                state.get("environment_config") != env.cfg or
                state.get("requested_moth_mode") != args.moth_mode or
                not math.isclose(state.get("initial_alpha", .1), args.initial_alpha, rel_tol=1e-12) or
                state.get("observation_frame", "world") != args.observation_frame):
            parser.error("incompatible trainer/physics/reward/opponent configuration; start a separately named run")
        if state["seed"] != args.seed:
            parser.error("checkpoint seed mismatch")
        bat.load_state_dict(state["bat"])
        moth.load_state_dict(state["moth"])
        critic.load_state_dict(state["critic"])
        critic_target.load_state_dict(state["critic_tgt"])
        log_alpha = torch.tensor(state["log_alpha"], requires_grad=True)
        actor_opt.load_state_dict(state["actor_opt"])
        critic_opt.load_state_dict(state["critic_opt"])
        alpha_opt = torch.optim.Adam([log_alpha], lr=SAC_LEARNING_RATE)
        alpha_opt.load_state_dict(state["alpha_opt"])
        episodes = state["episodes"]
        start = int(state["next_episode"])
        completed_updates = int(state.get("completed_updates", 0))
        env.load_novelty_state(state.get("novelty_state", {}))
        saved_buffer = state.get("replay_buffer")
        if saved_buffer is not None:
            buffer.load_state_dict(saved_buffer)
            print(f"Restored replay buffer: {len(buffer)} transitions in "
                  f"{buffer.episode_count} episodes.")
        else:
            print("Checkpoint has no replay buffer; starting with an empty buffer.")
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        moth_action_rng.bit_generator.state = state.get(
            "moth_action_rng", moth_action_rng.bit_generator.state)
        sample_rng.bit_generator.state = state.get("sample_rng", sample_rng.bit_generator.state)
        print(f"Resumed from episode {start}, {completed_updates} updates so far.")

    live_stream = None
    if not args.no_live:
        LIVE_FRAME_STREAM.parent.mkdir(parents=True, exist_ok=True)
        live_stream = LIVE_FRAME_STREAM.open("a" if args.resume else "w", encoding="utf-8")

    def stop(_signum, _frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    progress_path = args.output.with_name(f"{args.output.stem}_reward_progress.json")
    progress = {
        "status": "running", "seed": args.seed, "training_roles": ["bat"],
        "trainer_version": TRAINER_VERSION, "physics_version": PHYSICS_VERSION,
        "environment_config": env.cfg.copy(), "requested_moth_mode": args.moth_mode,
        "initial_alpha": args.initial_alpha,
        "observation_frame": args.observation_frame,
        "moth_opponent_checkpoint": str(opponent_path),
        "moth_opponent_probabilities": {"learned": 0.5, "random": 0.3, "stationary": 0.2},
        "sac": {"learning_rate": SAC_LEARNING_RATE, "gamma": SAC_GAMMA,
                "tau": SAC_TAU, "warmup_episodes": SAC_WARMUP_EPISODES},
        "completed_updates": completed_updates,
        "episodes": [{"episode": ep["episode"], "caught": ep["caught"],
            "episode_reward": ep.get("episode_reward"),
            "avg_reward_last_100": ep.get("avg_reward_last_100")} for ep in episodes],
    }

    for episode_index in range(start, args.episodes):
        seed = args.seed + episode_index
        env.reset(seed=seed)
        observations = {role: env.observe(role) for role in models}
        moth_mode = (random.choices(("learned", "random", "stationary"),
                                   weights=(0.5, 0.3, 0.2), k=1)[0]
                     if args.moth_mode == "mix" else args.moth_mode)
        env.cfg["stationary_moth"] = moth_mode == "stationary"
        hidden = {role: model.initial_hidden().detach() for role, model in models.items()}
        trajectory = {key: [] for key in ("obs", "actions", "rewards", "dones", "next_obs",
                                           "next_own", "opponent", "visible", "occluded")}
        moth_opponent_states = []
        moth_occluded = []
        update_losses = []
        visual_records = []
        reward_totals = {"bat": 0.0, "moth": 0.0}
        last_info = {}

        while True:
            raw_actions = {}
            sampled_outputs = {}
            for role, model in models.items():
                model_observation = (legacy_moth_observation(observations[role])
                                     if role == "moth" else bat_policy_observation(
                                         observations[role], args.observation_frame))
                observation = torch.as_tensor(model_observation, dtype=torch.float32)
                with torch.no_grad():
                    output = model.forward_sequence(observation.reshape(1, 1, -1), hidden[role])
                    mean = output.mean[:, 0]
                    distribution = Normal(mean, model.log_standard_deviation.exp().expand_as(mean))
                    action = distribution.sample()
                hidden[role] = output.hidden.detach()
                sampled_outputs[role] = output
                sampled_action = action[0].cpu().numpy()
                raw_actions[role] = (np.tanh(sampled_action) if role == "bat"
                                     else np.clip(sampled_action, -1.0, 1.0))
            if moth_mode == "random":
                raw_actions["moth"] = random_action(moth_action_rng)
            elif moth_mode == "stationary":
                raw_actions["moth"] = np.asarray((0.0, 0.0, -1.0), dtype=np.float32)

            opponent_target = _opponent_state(env, "bat")
            moth_opponent_states.append(_opponent_state(env, "moth"))
            moth_occluded.append(bool(observations["moth"][-1] > 0.5))
            next_observations, rewards, done, _, info = env.step_joint(raw_actions)
            # Repertoire novelty is diagnostic only: its changing archive makes
            # old replay rewards inconsistent with the current reward function.
            rewards = {role: float(info[f"{role}_reward"]) for role in models}
            for role in models:
                reward_totals[role] += float(rewards[role])
            trajectory["obs"].append(bat_policy_observation(observations["bat"], args.observation_frame))
            trajectory["actions"].append(raw_actions["bat"].copy())
            trajectory["rewards"].append(float(rewards["bat"]))
            trajectory["dones"].append(float(done))
            trajectory["next_obs"].append(bat_policy_observation(next_observations["bat"], args.observation_frame))
            trajectory["next_own"].append(np.concatenate((env.bat.position, env.bat.velocity)).copy())
            trajectory["opponent"].append(opponent_target)
            trajectory["visible"].append(bool(observations["bat"][9] > 0.5))
            trajectory["occluded"].append(bool(observations["bat"][-1] > 0.5))

            visual_records.append({role: {
                "attention": sampled_outputs[role].attention_weights[0, 0].cpu().tolist(),
                "workspace": sampled_outputs[role].workspace_weights[0, 0].cpu().tolist(),
                "belief": sampled_outputs[role].belief_state[0, 0].cpu().tolist()}
                for role in models} | {
                    "bat_alignment": info["bat_alignment"],
                    "bat_alignment_reward": info["bat_alignment_reward"],
                    "bat_aligned_thrust_reward": info["bat_aligned_thrust_reward"]})
            live_record = {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "phase": "9-SAC", "episode": episode_index + 1, "tick": env.tick,
                "caught": bool(info["caught"]), "jaw_distance": float(info["jaw_distance"]),
                "bat_sonar_hit": info["bat_alignment"] is not None,
                "bat_alignment": info["bat_alignment"],
                "bat_alignment_reward": info["bat_alignment_reward"],
                "bat_aligned_thrust_reward": info["bat_aligned_thrust_reward"],
                "jaw_contact_fraction": info["jaw_contact_fraction"],
                "jaw_contact_position": info["jaw_contact_position"],
                "sonar_occluded": bool(info["sonar_occluded"]),
                "vision_occluded": bool(info["vision_occluded"]),
                "attention_weights": {role: sampled_outputs[role].attention_weights[0, 0].cpu().tolist()
                                       for role in models},
                "workspace_weights": {role: dict(zip(SPECIALIST_NAMES,
                    sampled_outputs[role].workspace_weights[0, 0].cpu().tolist()))
                    for role in models},
                "workspace_dominant": workspace_dominants_by_role({role: dict(zip(
                    SPECIALIST_NAMES, sampled_outputs[role].workspace_weights[0, 0].cpu().tolist()))
                    for role in models}),
                "self_model_predicted_position": {role:
                    sampled_outputs[role].predicted_next_state[0, 0, :3].cpu().tolist()
                    for role in models},
                "belief_state_estimated_position": {role:
                    sampled_outputs[role].belief_state[0, 0, :3].cpu().tolist()
                    for role in models},
                "bat": {"position": info["bat_position"], "velocity": info["bat_velocity"],
                        "sonar_radius": float(env.cfg["sonar_range"]),
                        "jaw_position": info["mouth_position"]},
                "moth": {"position": info["moth_position"], "velocity": info["moth_velocity"],
                         "flame_position": info["flame_position"]},
                "environment": {"obstacles": [{"x": float(item["xy"][0]),
                    "y": float(item["xy"][1]), "angle": float(np.arctan2(item["branch"][1],
                    item["branch"][0]))} for item in env.obstacles]},
            }
            if live_stream is not None:
                atomic_json(DATA / "live_state.json", live_record)
                live_stream.write(json.dumps(live_record, separators=(",", ":")) + "\n")
                live_stream.flush()
            observations = next_observations
            last_info = info
            if done:
                break

        arrays = {key: np.asarray(value, dtype=np.float32)
                  for key, value in trajectory.items() if key not in ("visible", "occluded")}
        arrays["visible"] = np.asarray(trajectory["visible"], dtype=bool)
        arrays["occluded"] = np.asarray(trajectory["occluded"], dtype=bool)
        buffer.push(arrays)

        sac_losses = []
        if buffer.episode_count >= SAC_WARMUP_EPISODES:
            for _ in range(SAC_UPDATES_PER_EP):
                if len(buffer) >= SAC_BATCH_EPISODES * SAC_SEGMENT_LEN:
                    loss = _sac_update(bat, critic, critic_target, log_alpha,
                        actor_opt, critic_opt, alpha_opt, buffer, target_entropy, sample_rng)
                    sac_losses.append(loss)
                    update_losses.append(loss)
                    completed_updates += 1

        episode_number = episode_index + 1
        avg_rewards = {role: float(np.mean(
            [ep["episode_reward"][role] for ep in episodes[-99:]] + [reward_totals[role]]))
            for role in models}
        episode_record = {
            "episode": episode_number, "seed": seed, "steps": len(trajectory["obs"]),
            "hunt_length": len(trajectory["obs"]), "caught": bool(last_info["caught"]),
            "survived": not bool(last_info["caught"]), "moth_opponent_mode": moth_mode,
            "episode_reward": reward_totals, "avg_reward_last_100": avg_rewards,
            "sonar_occlusion_rate": last_info["occlusion_rate"]["bat"],
            "vision_occlusion_rate": last_info["occlusion_rate"]["moth"],
            "sac_losses": sac_losses, "completed_updates": completed_updates,
            "replay_buffer_size": len(buffer),
            "attention_weights_by_role": {role: [tick[role]["attention"] for tick in visual_records]
                                           for role in models},
            "workspace_weights_by_role": {role: [tick[role]["workspace"] for tick in visual_records]
                                            for role in models},
            "bat_alignment_by_tick": [tick["bat_alignment"] for tick in visual_records],
            "bat_alignment_reward_by_tick": [tick["bat_alignment_reward"] for tick in visual_records],
            "bat_aligned_thrust_reward_by_tick": [tick["bat_aligned_thrust_reward"]
                                                   for tick in visual_records],
            "belief_estimates_by_role": {role: [tick[role]["belief"] for tick in visual_records]
                                          for role in models},
            "opponent_states_by_role": {"bat": trajectory["opponent"],
                                        "moth": moth_opponent_states},
            "occlusion_by_role": {"bat": trajectory["occluded"], "moth": moth_occluded},
            "belief_mse_when_visible": {
                "bat": float(np.mean([loss["belief_mse"] for loss in update_losses]))
                       if update_losses else None,
                "moth": None},
        }
        episodes.append(episode_record)

        alpha_value = float(log_alpha.exp().detach())
        loss_summary = (f" critic {sac_losses[-1]['critic_loss']:.3f}"
                        f" actor {sac_losses[-1]['actor_loss']:.3f} alpha {alpha_value:.3f}"
                        if sac_losses else " (warmup)")
        print(f"SAC bat | ep {episode_number}/{args.episodes} | moth {moth_mode} | "
              f"{'CAUGHT' if last_info['caught'] else 'timeout'} | "
              f"reward {reward_totals['bat']:+.3f} | avg/100 {avg_rewards['bat']:+.3f} | "
              f"buf {len(buffer)}{loss_summary}", flush=True)

        progress["completed_updates"] = completed_updates
        progress["episodes"].append({"episode": episode_number, "caught": episode_record["caught"],
            "episode_reward": reward_totals, "moth_opponent_mode": moth_mode,
            "avg_reward_last_100": avg_rewards})
        atomic_json(progress_path, progress)
        if (episode_number % CHECKPOINT_INTERVAL_EPISODES == 0
                or episode_number >= args.episodes or STOP_REQUESTED):
            checkpoint_payload = {
            "trainer_version": TRAINER_VERSION, "physics_version": PHYSICS_VERSION,
            "environment_config": progress["environment_config"],
            "requested_moth_mode": args.moth_mode,
            "initial_alpha": args.initial_alpha,
            "observation_frame": args.observation_frame,
            "seed": args.seed, "next_episode": episode_number, "episodes": episodes,
            "bat": bat.state_dict(), "moth": moth.state_dict(),
            "critic": critic.state_dict(), "critic_tgt": critic_target.state_dict(),
            "log_alpha": float(log_alpha.detach()), "actor_opt": actor_opt.state_dict(),
            "critic_opt": critic_opt.state_dict(), "alpha_opt": alpha_opt.state_dict(),
            "completed_updates": completed_updates, "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "moth_action_rng": moth_action_rng.bit_generator.state,
            "sample_rng": sample_rng.bit_generator.state,
            "replay_buffer": buffer.state_dict(),
            "novelty_state": env.novelty_state(),
            }
            _atomic_torch_save(args.checkpoint, checkpoint_payload)
            if episode_number % SNAPSHOT_INTERVAL_EPISODES == 0 or episode_number >= args.episodes:
                snapshot = args.checkpoint.with_name(f"{args.checkpoint.stem}_ep{episode_number:04d}.pt")
                _atomic_torch_save(snapshot, checkpoint_payload)
        if STOP_REQUESTED:
            break

    if live_stream is not None:
        live_stream.close()
    finished = not STOP_REQUESTED and len(episodes) >= args.episodes
    _write_outputs(episodes, models, args.seed, finished, args.output, str(opponent_path),
                   progress["environment_config"], args.moth_mode, args.initial_alpha, args.observation_frame)
    progress["status"] = "complete" if finished else "stopped"
    atomic_json(progress_path, progress)


if __name__ == "__main__":
    main()
