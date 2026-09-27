"""Episode-synchronous PPO self-play: both agents update after every shared hunt."""

from __future__ import annotations

import argparse
import json
import signal
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor

from .novelty import NOVELTY_WARNING_RATE, NoveltyEnv
from .train import RESULTS
from .training_runtime import (DATA, TRAINING_MAX_THREADS, TickPacer, TrainingRuntime,
                               atomic_json, latest_full_checkpoint, load_full_policies)


STOP = threading.Event()


def _collect_action(model, observation):
    obs = np.asarray(observation, dtype=np.float32)[None, :]
    with torch.no_grad():
        actions, values, log_probs = model.policy(obs_as_tensor(obs, model.device))
    raw = actions.cpu().numpy()
    return raw, np.clip(raw, -1.0, 1.0), values, log_probs


def _new_rollout(model, env, capacity):
    return RolloutBuffer(capacity, env.observation_space, env.action_space,
                         device=model.device, gae_lambda=model.gae_lambda,
                         gamma=model.gamma, n_envs=1)


def _update_after_episode(model, buffer, episode_length, total_budget):
    buffer.compute_returns_and_advantage(torch.zeros((1,), device=model.device),
                                         np.ones(1, dtype=bool))
    model._total_timesteps = max(model.num_timesteps + 1, total_budget)
    model._update_current_progress_remaining(model.num_timesteps, model._total_timesteps)
    model.rollout_buffer = buffer
    model.train()


def _summary(episodes):
    last = episodes[-100:]
    if not last:
        return {"episodes": 0}
    return {"episodes": len(episodes),
            "catch_rate_last_100": float(np.mean([ep["caught"] for ep in last])),
            "survival_rate_last_100": float(np.mean([not ep["caught"] for ep in last])),
            "novel_behavior_rate_last_100": float(np.mean([ep["novel_behavior"] for ep in last])),
            "average_hunt_length_last_100": float(np.mean([ep["hunt_length"] for ep in last]))}


def train_joint_episode(models, env, rollout_envs, seed, total_budget,
                        on_tick=None, pace=None):
    """Collect one shared hunt, then update both PPO policies from that hunt."""
    env.reset(seed=seed)
    observations = {role: env.observe(role) for role in models}
    states = observations.copy()
    transitions = {role: [] for role in models}
    starts = {role: True for role in models}
    episode_length = 0
    while True:
        sampled = {role: _collect_action(models[role], observations[role]) for role in models}
        actions = {role: sample[1][0] for role, sample in sampled.items()}
        observations, rewards, done, _, info = env.step_joint(actions)
        episode_length += 1
        for role in models:
            raw, _, values, log_probs = sampled[role]
            transitions[role].append((np.asarray(states[role], dtype=np.float32)[None, :],
                raw.copy(), float(rewards[role]), starts[role], values, log_probs))
            models[role].num_timesteps += 1
            starts[role] = False
        if on_tick:
            on_tick(info)
        if pace and not pace():
            # Finish this real episode without further sleeping before checkpointing.
            pace = None
        if done:
            break
        states = observations.copy()
    for role, model in models.items():
        buffer = _new_rollout(model, rollout_envs[role].unwrapped, episode_length)
        for state, action, reward, episode_start, value, log_prob in transitions[role]:
            buffer.add(state, action, np.asarray([reward], dtype=np.float32),
                       np.asarray([episode_start], dtype=bool), value, log_prob)
        _update_after_episode(model, buffer, episode_length, total_budget)
    return episode_length, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--no-pacing", action="store_true")
    args = parser.parse_args()

    checkpoint = latest_full_checkpoint()
    prior = json.loads(checkpoint.read_text()) if checkpoint else None
    seed = int(args.seed if args.seed is not None else prior.get("seed", 23) if prior else 23)
    torch.set_num_threads(TRAINING_MAX_THREADS)

    shared_env = NoveltyEnv("bat")
    bat_env, moth_env = Monitor(NoveltyEnv("bat")), Monitor(NoveltyEnv("moth"))
    envs = {"bat": bat_env, "moth": moth_env}
    if prior:
        models, _ = load_full_policies(checkpoint, envs)
        bat, moth = models["bat"], models["moth"]
        repertoire_states = prior.get("behavioral_repertoires", {})
        sources = [repertoire_states.get(role, repertoire_states.get(f"{role}_env", {}))
                   for role in ("bat", "moth")]
        entries = {role: source.get("repertoire", {}).get("entries", {}).get(role, [])
                   for role, source in zip(("bat", "moth"), sources)}
        shared_env.load_novelty_state({"repertoire": {
            "capacity": 50, "threshold": 0.4, "entries": entries}})
        start_step = max(int(prior.get("policy_training_steps", {}).get(role, 0))
                         for role in ("bat", "moth"))
        print(f"Resumed step {start_step} from full checkpoint {checkpoint.name}.", flush=True)
    else:
        gate = json.loads((RESULTS / "phase0_learning_curve.json").read_text())
        if not gate["gate"]["passed"]:
            parser.error("Phase 0 learning gate has not passed")
        bat = PPO.load(RESULTS / "phase0_bat_ppo.zip", env=bat_env, device="cpu")
        moth = PPO.load(RESULTS / "phase0_moth_ppo.zip", env=moth_env, device="cpu")
        start_step = max(bat.num_timesteps, moth.num_timesteps)

    models = {"bat": bat, "moth": moth}
    for model in models.values():
        model._setup_learn(total_timesteps=1, reset_num_timesteps=False)
    trackers = {role: SimpleNamespace(episodes=[]) for role in models}
    metrics = trackers
    runtime = TrainingRuntime(models, {"bat": shared_env, "moth": shared_env}, metrics,
                              seed, [], args.output or RESULTS / "selfplay_latest.json",
                              started_steps=start_step)
    runtime.training_mode = "episode-synchronous PPO self-play"
    runtime.selfplay_steps = int(prior.get("selfplay_steps", 0)) if prior and prior.get("training_mode") == runtime.training_mode else 0
    if prior and prior.get("training_mode") == runtime.training_mode:
        for role in models:
            trackers[role].episodes = list(prior.get("roles", {}).get(role, {}).get("episodes", []))
    episode_number = max((len(tracker.episodes) for tracker in trackers.values()), default=0)
    pacer = TickPacer(enabled=not args.no_pacing)
    stop_after_episode = False

    def request_stop(_signum, _frame):
        STOP.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    selfplay_steps = runtime.selfplay_steps
    total_budget = (2**63 - 1 if args.continuous else
                    start_step + args.episodes * shared_env.cfg["max_ticks"])
    while args.continuous or episode_number < args.episodes:
        if STOP.is_set():
            break
        def publish(info):
            runtime.steps_per_role = min(models["bat"].num_timesteps, models["moth"].num_timesteps)
            runtime.write_live_state("bat", info)
        episode_length, final_info = train_joint_episode(
            models, shared_env, envs, seed + episode_number, total_budget,
            on_tick=publish, pace=lambda: pacer.pace(STOP.is_set))
        stop_after_episode = STOP.is_set()

        episode_number += 1
        selfplay_steps += episode_length
        runtime.selfplay_steps = selfplay_steps
        caught = bool(final_info.get("caught", False))
        for role in models:
            repertoire = final_info.get("behavior_metrics", {}).get(role, {})
            trackers[role].episodes.append({"episode": episode_number,
                "step": int(models[role].num_timesteps), "caught": caught,
                "survived": not caught, "hunt_length": episode_length,
                "novel_behavior": bool(repertoire.get("novel_behavior", False)),
                "novelty_distance": float(repertoire.get("novelty_distance", 0.0))})
        print(f"self-play: bat step {selfplay_steps}, moth step {selfplay_steps}", flush=True)
        for role in models:
            recent = trackers[role].episodes[-100:]
            novelty_rate = float(np.mean([ep["novel_behavior"] for ep in recent])) if recent else 0.0
            if len(recent) >= 20 and novelty_rate > NOVELTY_WARNING_RATE:
                print(f"WARNING: {role} corrected novel behavior rate {novelty_rate:.3f} > "
                      f"{NOVELTY_WARNING_RATE:.0%}; inspect repertoire threshold/initial growth.", flush=True)
        runtime.maybe_checkpoint()
        if episode_number % 100 == 0 or stop_after_episode:
            atomic_json(runtime.output_path, {"phase": 4, "training_mode": "episode-synchronous PPO self-play",
                "seed": seed, "steps_per_agent": {role: int(models[role].num_timesteps)
                                                   for role in models},
                "selfplay_steps": selfplay_steps,
                "roles": {role: {"summary": _summary(trackers[role].episodes),
                                 "episodes": trackers[role].episodes} for role in models},
                "behavioral_repertoires": shared_env.novelty_state()})
        if stop_after_episode:
            break

    checkpoint_path = runtime.save_full_checkpoint("shutdown" if STOP.is_set() else "completion")
    atomic_json(runtime.output_path, {"phase": 4, "training_mode": "episode-synchronous PPO self-play",
        "seed": seed, "steps_per_agent": {role: int(models[role].num_timesteps) for role in models},
        "selfplay_steps": selfplay_steps,
        "roles": {role: {"summary": _summary(trackers[role].episodes),
                         "episodes": trackers[role].episodes} for role in models},
        "behavioral_repertoires": shared_env.novelty_state(), "final_checkpoint": str(checkpoint_path)})
    print(f"Full checkpoint saved: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
