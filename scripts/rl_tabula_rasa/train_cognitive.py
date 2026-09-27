"""Episode-synchronous recurrent PPO for the Phase 9 cognitive policy."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Normal

from .baselines import random_action
from .cognitive import (BELIEF_LOSS_WEIGHT, COGNITIVE_ARCH_ENABLED,
                        SELF_MODEL_LOSS_WEIGHT, CognitiveActorCritic,
                        SPECIALIST_NAMES, added_parameters,
                        workspace_dominants_by_role)
from .environment_config import OBSTACLE_SEED
from .novelty import NoveltyEnv
from .training_runtime import DATA, TRAINING_MAX_THREADS, atomic_json


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CHECKPOINT = DATA / "cognitive_checkpoint.pt"
PPO_CLIP = 0.1
PPO_EPOCHS = 5
PPO_LEARNING_RATE = 1e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_VALUE_WEIGHT = 0.5
PPO_ENTROPY_WEIGHT = 0.05
PPO_UPDATE_EPISODES = 8
STOP_REQUESTED = False


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


def _visible(observation):
    return observation[..., 9] > 0.5


def _returns_and_advantages(rewards, values):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for index in reversed(range(len(rewards))):
        next_value = 0.0 if index == len(rewards) - 1 else values[index + 1]
        delta = rewards[index] + PPO_GAMMA * next_value - values[index]
        running = delta + PPO_GAMMA * PPO_GAE_LAMBDA * running
        advantages[index] = running
    return advantages, advantages + values


def _update(model, optimizer, episodes):
    advantages = np.concatenate([episode["advantages"] for episode in episodes])
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    batches, offset = [], 0
    for episode in episodes:
        length = len(episode["advantages"])
        batches.append({
            "obs": torch.as_tensor(np.asarray(episode["obs"]), dtype=torch.float32).unsqueeze(0),
            "actions": torch.as_tensor(np.asarray(episode["actions"]), dtype=torch.float32).unsqueeze(0),
            "old_logp": torch.as_tensor(np.asarray(episode["old_logp"]), dtype=torch.float32).unsqueeze(0),
            "returns": torch.as_tensor(np.asarray(episode["returns"]), dtype=torch.float32).unsqueeze(0),
            "advantages": torch.as_tensor(advantages[offset:offset + length], dtype=torch.float32).unsqueeze(0),
            "next_own": torch.as_tensor(np.asarray(episode["next_own"]), dtype=torch.float32).unsqueeze(0),
            "opponent": torch.as_tensor(np.asarray(episode["opponent"]), dtype=torch.float32).unsqueeze(0),
            "visible": torch.as_tensor(np.asarray(episode["visible"]), dtype=torch.bool).unsqueeze(0),
        })
        offset += length
    losses = []
    model.train()
    for _ in range(PPO_EPOCHS):
        outputs, distributions = [], []
        logps, old_logps, batch_returns, batch_advantages = [], [], [], []
        next_own, opponent, visible = [], [], []
        for batch in batches:
            output = model.forward_sequence(batch["obs"])
            distribution = Normal(output.mean,
                                  model.log_standard_deviation.exp().expand_as(output.mean))
            outputs.append(output)
            distributions.append(distribution)
            logps.append(distribution.log_prob(batch["actions"]).sum(dim=-1).reshape(-1))
            old_logps.append(batch["old_logp"].reshape(-1))
            batch_returns.append(batch["returns"].reshape(-1))
            batch_advantages.append(batch["advantages"].reshape(-1))
            next_own.append(batch["next_own"].reshape(-1, 6))
            opponent.append(batch["opponent"].reshape(-1, 6))
            visible.append(batch["visible"].reshape(-1))
        logp = torch.cat(logps)
        ratio = (logp - torch.cat(old_logps)).exp()
        advantages = torch.cat(batch_advantages)
        surrogate = torch.minimum(ratio * advantages,
                                  torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantages)
        policy_loss = -surrogate.mean()
        value_loss = torch.nn.functional.mse_loss(torch.cat([o.value.reshape(-1) for o in outputs]),
                                                   torch.cat(batch_returns))
        entropy = torch.cat([d.entropy().sum(dim=-1).reshape(-1)
                             for d in distributions]).mean()
        self_loss = torch.nn.functional.mse_loss(
            torch.cat([o.predicted_next_state.reshape(-1, 6) for o in outputs]),
            torch.cat(next_own))
        belief_error = (torch.cat([o.belief_state.reshape(-1, 6) for o in outputs])
                        - torch.cat(opponent)).square().mean(dim=-1)
        visible = torch.cat(visible)
        belief_loss = belief_error[visible].mean() if visible.any() else belief_error.new_zeros(())
        total = (policy_loss + PPO_VALUE_WEIGHT * value_loss - PPO_ENTROPY_WEIGHT * entropy
                 + SELF_MODEL_LOSS_WEIGHT * self_loss + BELIEF_LOSS_WEIGHT * belief_loss)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        losses.append({"total": float(total.detach()), "policy": float(policy_loss.detach()),
                       "value": float(value_loss.detach()), "self_model_mse": float(self_loss.detach()),
                       "belief_mse": float(belief_loss.detach()), "entropy": float(entropy.detach())})
    model.eval()
    return {name: float(np.mean([item[name] for item in losses])) for name in losses[0]}


def _analysis(episodes, key, names_by_role):
    result = {}
    for role in ("bat", "moth"):
        selected = [episode for episode in episodes if episode.get(key)]
        if not selected:
            result[role] = {}
            continue
        width = len(names_by_role[role])
        bins = {"early": [], "mid": [], "final_20_ticks": []}
        for episode in selected:
            values = np.zeros((episode["hunt_length"], width), dtype=np.float64)
            source = episode[key][role]
            values[:min(len(source), len(values))] = source[:len(values)]
            n = len(values)
            bins["early"].append(values[:max(1, n // 3)].mean(axis=0))
            bins["mid"].append(values[n // 3:max(n // 3 + 1, 2 * n // 3)].mean(axis=0))
            bins["final_20_ticks"].append(values[max(0, n - 20):].mean(axis=0))
        result[role] = {phase: {name: float(np.mean([row[i] for row in phase_values]))
                                for i, name in enumerate(names_by_role[role])}
                        for phase, phase_values in bins.items()}
    return result


def _write_outputs(episodes, models, seed, finished, output_path, training_roles,
                   moth_opponent_checkpoint):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_agent_params = {role: added_parameters(role, model.observation_dim)
                        for role, model in models.items()}
    opponent_mix = {}
    for mode in ("learned", "random", "stationary"):
        selected = [episode for episode in episodes
                    if episode.get("moth_opponent_mode", "learned") == mode]
        opponent_mix[mode] = {"episodes": len(selected),
            "fraction": len(selected) / len(episodes) if episodes else None,
            "catch_rate": float(np.mean([ep["caught"] for ep in selected])) if selected else None}
    payload = {"phase": 9, "architecture": "cognitive recurrent PPO",
               "cognitive_arch_enabled": COGNITIVE_ARCH_ENABLED,
               "seed": seed, "completed_episodes": len(episodes),
               "training_roles": list(training_roles),
               "frozen_moth_opponent_checkpoint": moth_opponent_checkpoint,
               "moth_opponent_mix": opponent_mix,
               "ppo": {"clip": PPO_CLIP, "learning_rate": PPO_LEARNING_RATE,
                       "entropy_weight": PPO_ENTROPY_WEIGHT,
                       "episodes_per_update": PPO_UPDATE_EPISODES},
               "parameters": per_agent_params,
               "total_added_parameters": sum(item["added_parameters"]
                                               for item in per_agent_params.values()),
               "episodes": episodes, "status": "complete" if finished else "running"}
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
                {"phase": 9, "attention_by_hunt_phase": _analysis(episodes,
                    "attention_weights_by_role", attention_names)})
    atomic_json(output_path.with_name(f"{output_path.stem}_workspace_analysis.json"),
                {"phase": 9, "workspace_dominance_by_hunt_phase": _analysis(episodes,
                    "workspace_weights_by_role", workspace_names)})
    belief = {}
    for role in ("bat", "moth"):
        errors = []
        count = 0
        for episode in episodes:
            estimate = np.asarray(episode["belief_estimates_by_role"][role], dtype=np.float32)
            actual = np.asarray(episode["opponent_states_by_role"][role], dtype=np.float32)
            occluded = np.asarray(episode["occlusion_by_role"][role], dtype=bool)
            count += int(occluded.sum())
            if occluded.any():
                errors.extend(np.linalg.norm(estimate[occluded] - actual[occluded], axis=-1).tolist())
        visible_losses = [ep["belief_mse_when_visible"][role] for ep in episodes
                          if ep.get("belief_mse_when_visible", {}).get(role) is not None]
        belief[role] = {"ticks_occluded": count,
            "mean_drift_distance_during_occlusion": float(np.mean(errors)) if errors else None,
            "mean_supervised_belief_mse": float(np.mean(visible_losses)) if visible_losses else None}
    atomic_json(output_path.with_name(f"{output_path.stem}_belief_analysis.json"), {"phase": 9, "belief_tracking": belief,
        "note": "Belief MSE is supervised only on visible ticks; drift analysis is computed from logged estimates on occluded ticks."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--moth-opponent-checkpoint", type=Path,
                        help="freeze this checkpoint's moth and train the bat against a 50/30/20 opponent mix")
    default_output = RESULTS / f"cognitive_architecture_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(TRAINING_MAX_THREADS)
    np.random.seed(args.seed)
    random.seed(args.seed)
    moth_action_rng = np.random.default_rng(args.seed + 901)
    env = NoveltyEnv("bat", config={"natural_environment": True,
        "obstacle_seed": args.seed if OBSTACLE_SEED is None else OBSTACLE_SEED})
    models = {role: CognitiveActorCritic(role, 19 if role == "bat" else 23)
              for role in ("bat", "moth")}
    if args.moth_opponent_checkpoint:
        opponent_path = args.moth_opponent_checkpoint.resolve()
        if not opponent_path.is_file():
            parser.error(f"moth opponent checkpoint not found: {opponent_path}")
        opponent_state = torch.load(opponent_path, map_location="cpu", weights_only=False)
        if "moth" not in opponent_state.get("models", {}):
            parser.error(f"checkpoint has no moth policy: {opponent_path}")
        models["moth"].load_state_dict(opponent_state["models"]["moth"])
    training_roles = ("bat",) if args.moth_opponent_checkpoint else tuple(models)
    optimizers = {role: torch.optim.Adam(models[role].parameters(), lr=PPO_LEARNING_RATE)
                  for role in training_roles}
    episodes = []
    pending_rollouts = {role: [] for role in training_roles}
    pending_episode_indices = []
    completed_updates = 0
    start = 0
    if args.resume and args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if state["seed"] != args.seed:
            parser.error("checkpoint run seed differs; choose its seed or move the checkpoint")
        if tuple(state.get("training_roles", models)) != training_roles:
            parser.error("checkpoint training roles differ from this run's opponent setup")
        expected_opponent = (str(args.moth_opponent_checkpoint.resolve())
                             if args.moth_opponent_checkpoint else None)
        if state.get("moth_opponent_checkpoint") != expected_opponent:
            parser.error("checkpoint moth-opponent source differs from this run")
        for role in models:
            models[role].load_state_dict(state["models"][role])
        for role in training_roles:
            optimizers[role].load_state_dict(state["optimizers"][role])
        episodes = state["episodes"]
        start = int(state["next_episode"])
        pending_rollouts = state.get("pending_rollouts", pending_rollouts)
        if set(pending_rollouts) != set(training_roles):
            parser.error("checkpoint has an incompatible pending PPO batch")
        pending_episode_indices = state.get("pending_episode_indices", [])
        completed_updates = int(state.get("completed_updates", 0))
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        moth_action_rng.bit_generator.state = state.get(
            "moth_action_rng", moth_action_rng.bit_generator.state)

    def stop(_signum, _frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    models = {role: model.eval() for role, model in models.items()}
    progress_path = args.output.with_name(f"{args.output.stem}_reward_progress.json")
    progress = {
        "status": "running", "seed": args.seed,
        "training_roles": list(training_roles),
        "moth_opponent_checkpoint": str(args.moth_opponent_checkpoint.resolve())
            if args.moth_opponent_checkpoint else None,
        "moth_opponent_probabilities": {"learned": 0.5, "random": 0.3,
                                        "stationary": 0.2}
            if args.moth_opponent_checkpoint else None,
        "ppo_update_episodes": PPO_UPDATE_EPISODES,
        "completed_updates": completed_updates,
        "ppo": {"clip": PPO_CLIP, "learning_rate": PPO_LEARNING_RATE,
                "entropy_weight": PPO_ENTROPY_WEIGHT},
        "reward_coefficients": {key: env.cfg[key] for key in (
            "bat_tick", "bat_distance_reduction", "moth_bat_distance_gain",
            "moth_flame_distance_reduction")},
        "entropy_weight": PPO_ENTROPY_WEIGHT,
        "episodes": [{"episode": ep["episode"], "caught": ep["caught"],
            "episode_reward": ep.get("episode_reward"),
            "avg_reward_last_100": ep.get("avg_reward_last_100")} for ep in episodes]}
    for episode_index in range(start, args.episodes):
        seed = args.seed + episode_index
        observations, _ = env.reset(seed=seed)
        observations = {role: env.observe(role) for role in models}
        moth_opponent_mode = (random.choices(("learned", "random", "stationary"),
            weights=(0.5, 0.3, 0.2), k=1)[0] if args.moth_opponent_checkpoint else "learned")
        env.cfg["stationary_moth"] = moth_opponent_mode == "stationary"
        hidden = {role: model.initial_hidden().detach() for role, model in models.items()}
        trajectory = {role: {"obs": [], "actions": [], "old_logp": [], "values": [],
                             "rewards": [], "next_own": [], "opponent": [],
                             "visible": [], "occluded": []} for role in models}
        visual_records = []
        last_info = {}
        reward_totals = {role: 0.0 for role in models}
        while True:
            sampled, raw_actions = {}, {}
            sampled_outputs = {}
            for role, model in models.items():
                obs_tensor = torch.as_tensor(observations[role], dtype=torch.float32)
                with torch.no_grad():
                    action, logp, value, output = model.act(obs_tensor, hidden[role])
                hidden[role] = output.hidden.detach()
                sampled_outputs[role] = output
                raw = action[0].cpu().numpy()
                sampled[role] = (raw, float(logp[0]), float(value[0]))
                raw_actions[role] = np.clip(raw, -1.0, 1.0)
            if moth_opponent_mode == "random":
                raw_actions["moth"] = random_action(moth_action_rng)
            elif moth_opponent_mode == "stationary":
                raw_actions["moth"] = np.asarray((0.0, 0.0, -1.0), dtype=np.float32)
            opponent_targets = {role: _opponent_state(env, role) for role in models}
            next_observations, rewards, done, _, info = env.step_joint(raw_actions)
            for role in models:
                reward_totals[role] += float(rewards[role])
            visual_records.append({role: {
                "attention": sampled_outputs[role].attention_weights[0, 0].cpu().tolist(),
                "workspace": sampled_outputs[role].workspace_weights[0, 0].cpu().tolist(),
                "belief": sampled_outputs[role].belief_state[0, 0].cpu().tolist()}
                for role in models})
            for role in training_roles:
                raw, logp, value = sampled[role]
                trajectory[role]["obs"].append(observations[role].copy())
                trajectory[role]["actions"].append(raw.copy())
                trajectory[role]["old_logp"].append(logp)
                trajectory[role]["values"].append(value)
                trajectory[role]["rewards"].append(float(rewards[role]))
                trajectory[role]["next_own"].append(np.concatenate((
                    getattr(env, role).position, getattr(env, role).velocity)).copy())
                trajectory[role]["opponent"].append(opponent_targets[role])
                trajectory[role]["visible"].append(bool(observations[role][9] > 0.5))
                trajectory[role]["occluded"].append(bool(observations[role][-1] > 0.5))
            atomic_json(DATA / "live_state.json", {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 9, "episode": episode_index + 1, "tick": env.tick,
                "active_training_role": "+".join(training_roles), "caught": bool(info["caught"]),
                "jaw_distance": float(info["jaw_distance"]),
                "jaw_contact_fraction": info["jaw_contact_fraction"],
                "jaw_contact_position": info["jaw_contact_position"],
                "sonar_occluded": bool(info["sonar_occluded"]),
                "vision_occluded": bool(info["vision_occluded"]),
                "attention_weights": {role: sampled_outputs[role].attention_weights[0, 0].cpu().tolist()
                                      for role in models},
                "workspace_weights": {role: dict(zip(SPECIALIST_NAMES,
                    sampled_outputs[role].workspace_weights[0, 0].cpu().tolist())) for role in models},
                "workspace_dominant": workspace_dominants_by_role({role: dict(zip(
                    SPECIALIST_NAMES, sampled_outputs[role].workspace_weights[0, 0].cpu().tolist()))
                    for role in models}),
                "self_model_predicted_position": {role:
                    sampled_outputs[role].predicted_next_state[0, 0, :3].cpu().tolist()
                    for role in models},
                "belief_state_estimated_position": {role:
                    sampled_outputs[role].belief_state[0, 0, :3].cpu().tolist() for role in models},
                "bat": {"position": info["bat_position"], "velocity": info["bat_velocity"],
                        "sonar_radius": float(env.cfg["sonar_range"]),
                        "jaw_position": info["mouth_position"]},
                "moth": {"position": info["moth_position"], "velocity": info["moth_velocity"],
                         "flame_position": info["flame_position"]},
                "environment": {"obstacles": [{"x": float(item["xy"][0]),
                    "y": float(item["xy"][1]), "angle": float(np.arctan2(item["branch"][1],
                    item["branch"][0]))} for item in env.obstacles]}})
            observations = next_observations
            last_info = info
            if done:
                break

        for role in training_roles:
            row = trajectory[role]
            advantages, returns = _returns_and_advantages(np.asarray(row["rewards"], dtype=np.float32),
                                                           np.asarray(row["values"], dtype=np.float32))
            row["advantages"], row["returns"] = advantages, returns
            pending_rollouts[role].append(row)
        episode_rewards = reward_totals
        episodes.append({"episode": episode_index + 1, "seed": seed,
            "steps": len(trajectory["bat"]["rewards"]),
            "hunt_length": len(trajectory["bat"]["rewards"]), "caught": bool(last_info["caught"]),
            "survived": not bool(last_info["caught"]),
            "moth_opponent_mode": moth_opponent_mode,
            "episode_reward": episode_rewards,
            "avg_reward_last_100": {role: float(np.mean([
                ep["episode_reward"][role] for ep in episodes[-99:]] + [episode_rewards[role]]))
                for role in models},
            "sonar_occlusion_rate": last_info["occlusion_rate"]["bat"],
            "vision_occlusion_rate": last_info["occlusion_rate"]["moth"],
            "attention_weights_by_role": {role: [tick[role]["attention"] for tick in visual_records]
                                           for role in models},
            "workspace_weights_by_role": {role: [tick[role]["workspace"] for tick in visual_records]
                                            for role in models},
            "belief_estimates_by_role": {role: [tick[role]["belief"] for tick in visual_records]
                                          for role in models},
            "opponent_states_by_role": {role: trajectory[role]["opponent"] for role in models},
            "occlusion_by_role": {role: [bool(v) for v in trajectory[role]["occluded"]]
                                  for role in models}})
        pending_episode_indices.append(len(episodes) - 1)
        batch_size = len(pending_episode_indices)
        should_update = (len(pending_episode_indices) >= PPO_UPDATE_EPISODES
                         or episode_index == args.episodes - 1 or STOP_REQUESTED)
        if should_update and pending_episode_indices:
            losses = {role: _update(models[role], optimizers[role], pending_rollouts[role])
                      for role in training_roles}
            for pending_index in pending_episode_indices:
                episode = episodes[pending_index]
                episode["ppo_update"] = completed_updates + 1
                episode["self_model_mse"] = {
                    role: losses[role]["self_model_mse"] if role in losses else None
                    for role in models}
                episode["belief_mse_when_visible"] = {
                    role: losses[role]["belief_mse"] if role in losses else None
                    for role in models}
                episode["policy_loss"] = {
                    role: losses[role]["policy"] if role in losses else None
                    for role in models}
            completed_updates += 1
            pending_rollouts = {role: [] for role in training_roles}
            pending_episode_indices = []
            progress["completed_updates"] = completed_updates
        mode_label = ("bat training / frozen-moth mix" if args.moth_opponent_checkpoint
                      else "self-play")
        print(f"{mode_label}: bat step {sum(ep['steps'] for ep in episodes)}, "
              f"episode {episode_index + 1}/{args.episodes} | "
              f"moth mode {moth_opponent_mode} | "
              f"PPO batch {batch_size}/{PPO_UPDATE_EPISODES}"
              f"{' updated' if should_update else ''} | "
              f"episode reward bat {episode_rewards['bat']:+.3f}, moth {episode_rewards['moth']:+.3f} | "
              f"avg/100 bat {episodes[-1]['avg_reward_last_100']['bat']:+.3f}, "
              f"moth {episodes[-1]['avg_reward_last_100']['moth']:+.3f}", flush=True)
        _atomic_torch_save(args.checkpoint, {"seed": args.seed, "next_episode": episode_index + 1,
            "episodes": episodes, "models": {role: model.state_dict() for role, model in models.items()},
            "optimizers": {role: optimizer.state_dict() for role, optimizer in optimizers.items()},
            "training_roles": training_roles,
            "moth_opponent_checkpoint": str(args.moth_opponent_checkpoint.resolve())
                if args.moth_opponent_checkpoint else None,
            "pending_rollouts": pending_rollouts,
            "pending_episode_indices": pending_episode_indices,
            "completed_updates": completed_updates,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "moth_action_rng": moth_action_rng.bit_generator.state,
            "torch_rng": torch.get_rng_state()})
        progress["episodes"].append({"episode": episodes[-1]["episode"],
            "caught": episodes[-1]["caught"], "episode_reward": episode_rewards,
            "moth_opponent_mode": moth_opponent_mode,
            "avg_reward_last_100": episodes[-1]["avg_reward_last_100"]})
        atomic_json(progress_path, progress)
        if STOP_REQUESTED:
            break
    finished = not STOP_REQUESTED and len(episodes) >= args.episodes
    _write_outputs(episodes, models, args.seed, finished, args.output, training_roles,
                   str(args.moth_opponent_checkpoint.resolve())
                   if args.moth_opponent_checkpoint else None)
    progress["status"] = "complete" if finished else "stopped"
    atomic_json(progress_path, progress)


if __name__ == "__main__":
    main()
