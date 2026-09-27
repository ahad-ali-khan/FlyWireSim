"""True simultaneous flat-SAC co-evolution for bat and moth.

The two agents act from the same pre-step state, receive the same shared
physics transition, and both update from that transition before the next
episode.  This is deliberately separate from the older PPO self-play runner:
it uses the verified flat SAC observation/action interface and checkpoints the
two learners independently.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from .env import HuntEnv, PHYSICS_VERSION
from .train_cognitive_sac import bat_policy_observation, legacy_moth_observation
from .training_runtime import atomic_json


BUFFER_SIZE = 20_000
BATCH_SIZE = 128
LEARNING_STARTS = 1_000
UPDATE_EVERY = 1


def _role_env(role: str, seed: int) -> HuntEnv:
    """Create an exact-space environment used by SB3 policy/replay objects."""
    env = HuntEnv(role=role, config={"natural_environment": True, "obstacle_seed": None})
    env.reset(seed=seed)
    return env


def _add_transition(model: SAC, obs, next_obs, action, reward, done, info):
    model.replay_buffer.add(
        np.asarray(obs, dtype=np.float32)[None, :],
        np.asarray(next_obs, dtype=np.float32)[None, :],
        np.asarray(action, dtype=np.float32)[None, :],
        np.asarray([reward], dtype=np.float32),
        np.asarray([done], dtype=bool),
        [info],
    )


def _maybe_update(model: SAC, steps: int, total_steps: int):
    if model.replay_buffer.size() < max(LEARNING_STARTS, BATCH_SIZE):
        return False
    model._current_progress_remaining = max(0.0, 1.0 - steps / max(1, total_steps))
    model.train(gradient_steps=1, batch_size=BATCH_SIZE)
    return True


def _save_pair(bat: SAC, moth: SAC, output: Path, status: str, payload: dict):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".coevolution-save-") as tmp:
        tmp = Path(tmp)
        bat_tmp, moth_tmp = tmp / "bat.zip", tmp / "moth.zip"
        bat.save(bat_tmp)
        moth.save(moth_tmp)
        os.replace(bat_tmp, output.with_suffix(".bat.zip"))
        os.replace(moth_tmp, output.with_suffix(".moth.zip"))
        bat.save_replay_buffer(tmp / "bat.replay.pkl")
        moth.save_replay_buffer(tmp / "moth.replay.pkl")
        os.replace(tmp / "bat.replay.pkl", output.with_suffix(".bat.replay.pkl"))
        os.replace(tmp / "moth.replay.pkl", output.with_suffix(".moth.replay.pkl"))
    payload = dict(payload, status=status)
    atomic_json(output, payload)


def _load_or_create(args, bat_env, moth_env):
    if args.resume:
        if not all(args.output.with_suffix(s).is_file() for s in (
                ".json", ".bat.zip", ".moth.zip", ".bat.replay.pkl", ".moth.replay.pkl")):
            raise SystemExit("resume requires JSON, both model zips, and both replay buffers")
        run = json.loads(args.output.read_text())
        if (run["seed"] != args.seed or run["physics_version"] != PHYSICS_VERSION or
                run.get("moth_stationary_warmup", 200) != args.moth_stationary_warmup):
            raise SystemExit("resume configuration differs from saved co-evolution run")
        bat = SAC.load(args.output.with_suffix(".bat.zip"), env=bat_env, device="cpu")
        moth = SAC.load(args.output.with_suffix(".moth.zip"), env=moth_env, device="cpu")
        bat.load_replay_buffer(args.output.with_suffix(".bat.replay.pkl"))
        moth.load_replay_buffer(args.output.with_suffix(".moth.replay.pkl"))
        bat.set_logger(configure(format_strings=[]))
        moth.set_logger(configure(format_strings=[]))
        return bat, moth, run

    if args.bat_checkpoint is None:
        raise SystemExit("--bat-checkpoint is required for a fresh co-evolution run")
    bat = SAC.load(args.bat_checkpoint, env=bat_env, device="cpu")
    moth = SAC("MlpPolicy", moth_env, seed=args.seed + 1, device="cpu",
                learning_rate=3e-4, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                learning_starts=LEARNING_STARTS, gamma=.99,
                ent_coef="auto_0.005", train_freq=1, gradient_steps=1,
                policy_kwargs={"net_arch": [128, 128]})
    # Direct calls to SAC.train bypass BaseAlgorithm.learn(), which normally
    # creates the logger for us.
    bat.set_logger(configure(format_strings=[]))
    moth.set_logger(configure(format_strings=[]))
    return bat, moth, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--bat-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--update-every", type=int, default=UPDATE_EVERY)
    parser.add_argument("--moth-stationary-warmup", type=int, default=200,
                        help="linearly release moth from stationary-like actions")
    args = parser.parse_args()
    if not args.resume and any(args.output.with_suffix(s).exists() for s in (
            ".json", ".bat.zip", ".moth.zip")):
        parser.error("refusing to overwrite an existing co-evolution run")
    if args.episodes < 1 or args.update_every < 1:
        parser.error("episodes and update-every must be positive")

    torch.set_num_threads(1)
    bat_env = _role_env("bat", args.seed)
    moth_env = _role_env("moth", args.seed + 1)
    bat, moth, previous = _load_or_create(args, bat_env, moth_env)
    start_episode = len(previous["episodes"]) if previous else 0
    if args.episodes <= start_episode:
        parser.error("--episodes must exceed completed episode count")
    total_steps = args.episodes * 70
    episodes = list(previous["episodes"]) if previous else []
    updates = int(previous.get("updates", 0)) if previous else 0
    transition_steps = int(previous.get("transition_steps", 0)) if previous else 0
    env = HuntEnv(config={"natural_environment": True, "obstacle_seed": None})
    try:
        for episode_index in range(start_episode, args.episodes):
            raw_bat, _ = env.reset(seed=args.seed + episode_index)
            bat_obs = bat_policy_observation(raw_bat)
            moth_obs = legacy_moth_observation(env.observe("moth"))
            bat_reward = moth_reward = 0.0
            caught = False
            last_info = {}
            while True:
                bat_action, _ = bat.predict(bat_obs, deterministic=False)
                moth_action, _ = moth.predict(moth_obs, deterministic=False)
                if args.moth_stationary_warmup:
                    release = float(np.clip(
                        episode_index / args.moth_stationary_warmup, 0.0, 1.0))
                    stationary_action = np.array([0.0, 0.0, -1.0], dtype=np.float32)
                    moth_action = ((1.0 - release) * stationary_action
                                   + release * np.asarray(moth_action, dtype=np.float32))
                next_raw, _, done, _, info = env.step_with_actions({
                    "bat": np.asarray(bat_action, dtype=np.float32),
                    "moth": np.asarray(moth_action, dtype=np.float32),
                })
                next_bat_obs = bat_policy_observation(next_raw)
                next_moth_obs = legacy_moth_observation(env.observe("moth"))
                bat_reward += float(info["bat_reward"])
                moth_reward += float(info["moth_reward"])
                _add_transition(bat, bat_obs, next_bat_obs, bat_action,
                                info["bat_reward"], done, info)
                _add_transition(moth, moth_obs, next_moth_obs, moth_action,
                                info["moth_reward"], done, info)
                transition_steps += 1
                if transition_steps % args.update_every == 0:
                    updates += int(_maybe_update(bat, transition_steps, total_steps))
                    updates += int(_maybe_update(moth, transition_steps, total_steps))
                bat_obs, moth_obs = next_bat_obs, next_moth_obs
                last_info = info
                if done:
                    caught = bool(info["caught"])
                    break
            episodes.append({
                "episode": episode_index + 1,
                "caught": caught,
                "hunt_length": int(last_info["hunt_length"]),
                "bat_reward": bat_reward,
                "moth_reward": moth_reward,
                "obstacle_count": int(last_info["obstacle_count"]),
                "bat_replay": bat.replay_buffer.size(),
                "moth_replay": moth.replay_buffer.size(),
            })
            print(f"self-play: bat step {episode_index + 1}, moth step {episode_index + 1}", flush=True)
            if (episode_index + 1) % 10 == 0:
                recent = episodes[-10:]
                print(f"coevolution episodes {episode_index + 1}, last10 catch "
                      f"{np.mean([e['caught'] for e in recent]):.1%}, "
                      f"bat reward {np.mean([e['bat_reward'] for e in recent]):+.3f}, "
                      f"moth reward {np.mean([e['moth_reward'] for e in recent]):+.3f}", flush=True)
            if (episode_index + 1) % args.save_every == 0:
                payload = {
                    "algorithm": "stable_baselines3_sac_true_simultaneous_coevolution",
                    "physics_version": PHYSICS_VERSION, "seed": args.seed,
                    "environment_config": env.cfg, "episodes": episodes,
                    "updates": updates, "transition_steps": transition_steps,
                    "bat_checkpoint": str(args.bat_checkpoint) if args.bat_checkpoint else None,
                    "update_every": args.update_every,
                    "moth_stationary_warmup": args.moth_stationary_warmup,
                }
                _save_pair(bat, moth, args.output, "running", payload)
    except KeyboardInterrupt:
        status = "stopped"
    else:
        status = "complete"
    payload = {
        "algorithm": "stable_baselines3_sac_true_simultaneous_coevolution",
        "physics_version": PHYSICS_VERSION, "seed": args.seed,
        "environment_config": env.cfg, "episodes": episodes,
        "updates": updates, "transition_steps": transition_steps,
        "bat_checkpoint": str(args.bat_checkpoint) if args.bat_checkpoint else None,
        "update_every": args.update_every,
        "moth_stationary_warmup": args.moth_stationary_warmup,
    }
    _save_pair(bat, moth, args.output, status, payload)
    print(f"coevolution {status}: {len(episodes)} episodes, "
          f"catch rate {np.mean([e['caught'] for e in episodes]):.1%}", flush=True)


if __name__ == "__main__":
    main()
