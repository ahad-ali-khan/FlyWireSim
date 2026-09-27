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
LEAGUE_POOL_SIZE = 5
HISTORICAL_MOTH_PROBABILITY = 0.20
LEAGUE_EVAL_EVERY = 100
LEAGUE_EVAL_SEED_START = 60000
LEAGUE_EVAL_SEEDS = 200
LEAGUE_STOP_THRESHOLD = 0.30
LEAGUE_STOP_PATIENCE = 2


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


def snapshot_policy(model: SAC) -> dict:
    """Return a CPU-only frozen policy snapshot suitable for a league pool."""
    return {key: value.detach().cpu().clone()
            for key, value in model.policy.state_dict().items()}


def append_league_snapshot(pool: list[dict], model: SAC, episode: int,
                           max_size: int = LEAGUE_POOL_SIZE) -> list[dict]:
    """Append a labelled snapshot and retain only the newest max_size entries."""
    updated = list(pool)
    updated.append({"episode": int(episode), "state_dict": snapshot_policy(model)})
    return updated[-max_size:]


def choose_league_opponent(pool: list[dict], rng: np.random.Generator,
                           historical_probability: float = HISTORICAL_MOTH_PROBABILITY):
    """Choose current or one frozen historical moth with the configured mixture."""
    if not pool or float(rng.random()) >= historical_probability:
        return "current", None
    index = int(rng.integers(0, len(pool)))
    return "historical", pool[index]


def _save_pair(bat: SAC, moth: SAC, output: Path, status: str, payload: dict,
               league_pool: list[dict]):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".coevolution-save-") as tmp:
        tmp = Path(tmp)
        bat_tmp, moth_tmp, league_tmp = tmp / "bat.zip", tmp / "moth.zip", tmp / "league.pt"
        bat.save(bat_tmp)
        moth.save(moth_tmp)
        torch.save({"pool": league_pool}, league_tmp)
        os.replace(bat_tmp, output.with_suffix(".bat.zip"))
        os.replace(moth_tmp, output.with_suffix(".moth.zip"))
        os.replace(league_tmp, output.with_suffix(".league.pt"))
        bat.save_replay_buffer(tmp / "bat.replay.pkl")
        moth.save_replay_buffer(tmp / "moth.replay.pkl")
        os.replace(tmp / "bat.replay.pkl", output.with_suffix(".bat.replay.pkl"))
        os.replace(tmp / "moth.replay.pkl", output.with_suffix(".moth.replay.pkl"))
    payload = dict(payload, status=status)
    atomic_json(output, payload)


def _load_or_create(args, bat_env, moth_env):
    if args.resume:
        if not all(args.output.with_suffix(s).is_file() for s in (
                ".json", ".bat.zip", ".moth.zip", ".bat.replay.pkl", ".moth.replay.pkl",
                ".league.pt")):
            raise SystemExit("resume requires JSON, both model zips, both replay buffers, and league pool")
        run = json.loads(args.output.read_text())
        if (run["seed"] != args.seed or run["physics_version"] != PHYSICS_VERSION or
                run.get("moth_stationary_warmup", 200) != args.moth_stationary_warmup or
                run.get("league_pool_size", LEAGUE_POOL_SIZE) != args.league_pool_size or
                run.get("historical_moth_probability", HISTORICAL_MOTH_PROBABILITY) != args.historical_moth_probability):
            raise SystemExit("resume configuration differs from saved co-evolution run")
        bat = SAC.load(args.output.with_suffix(".bat.zip"), env=bat_env, device="cpu")
        moth = SAC.load(args.output.with_suffix(".moth.zip"), env=moth_env, device="cpu")
        bat.load_replay_buffer(args.output.with_suffix(".bat.replay.pkl"))
        moth.load_replay_buffer(args.output.with_suffix(".moth.replay.pkl"))
        bat.set_logger(configure(format_strings=[]))
        moth.set_logger(configure(format_strings=[]))
        league = torch.load(args.output.with_suffix(".league.pt"), map_location="cpu",
                            weights_only=False)
        return bat, moth, run, league.get("pool", [])

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
    return bat, moth, None, []


def evaluate_current_pair(bat: SAC, moth: SAC, seeds: int = LEAGUE_EVAL_SEEDS,
                          seed_start: int = LEAGUE_EVAL_SEED_START) -> dict:
    """Evaluate the current pair without exploration on held-out layouts."""
    outcomes = []
    for seed in range(seed_start, seed_start + seeds):
        env = HuntEnv(config={"natural_environment": True, "obstacle_seed": None})
        raw, _ = env.reset(seed=seed)
        while True:
            bat_action, _ = bat.predict(bat_policy_observation(raw), deterministic=True)
            moth_action, _ = moth.predict(
                legacy_moth_observation(env.observe("moth")), deterministic=True)
            raw, _, done, _, info = env.step_with_actions({
                "bat": np.asarray(bat_action, dtype=np.float32),
                "moth": np.asarray(moth_action, dtype=np.float32),
            })
            if done:
                outcomes.append(bool(info["caught"]))
                break
    return {
        "seed_start": seed_start,
        "seeds": seeds,
        "catch_rate": float(np.mean(outcomes)),
        "catch_count": int(np.sum(outcomes)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--bat-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--update-every", type=int, default=UPDATE_EVERY)
    parser.add_argument("--league-pool-size", type=int, default=LEAGUE_POOL_SIZE)
    parser.add_argument("--historical-moth-probability", type=float,
                        default=HISTORICAL_MOTH_PROBABILITY)
    parser.add_argument("--eval-every", type=int, default=LEAGUE_EVAL_EVERY)
    parser.add_argument("--eval-seeds", type=int, default=LEAGUE_EVAL_SEEDS)
    parser.add_argument("--eval-seed-start", type=int, default=LEAGUE_EVAL_SEED_START)
    parser.add_argument("--stop-threshold", type=float, default=LEAGUE_STOP_THRESHOLD)
    parser.add_argument("--stop-patience", type=int, default=LEAGUE_STOP_PATIENCE)
    parser.add_argument("--moth-stationary-warmup", type=int, default=200,
                        help="linearly release moth from stationary-like actions")
    args = parser.parse_args()
    if not args.resume and any(args.output.with_suffix(s).exists() for s in (
            ".json", ".bat.zip", ".moth.zip")):
        parser.error("refusing to overwrite an existing co-evolution run")
    if (args.episodes < 1 or args.update_every < 1 or args.league_pool_size < 1 or
            args.eval_every < 1 or args.eval_seeds < 1 or args.stop_patience < 1 or
            not 0 <= args.historical_moth_probability <= 1):
        parser.error("episode, update, league, evaluation, and patience values must be positive; probability must be in [0, 1]")

    torch.set_num_threads(1)
    bat_env = _role_env("bat", args.seed)
    moth_env = _role_env("moth", args.seed + 1)
    bat, moth, previous, league_pool = _load_or_create(args, bat_env, moth_env)
    start_episode = len(previous["episodes"]) if previous else 0
    if args.episodes <= start_episode:
        parser.error("--episodes must exceed completed episode count")
    total_steps = args.episodes * 70
    episodes = list(previous["episodes"]) if previous else []
    evaluations = list(previous.get("evaluations", [])) if previous else []
    updates = int(previous.get("updates", 0)) if previous else 0
    transition_steps = int(previous.get("transition_steps", 0)) if previous else 0
    consecutive_passes = int(previous.get("consecutive_league_passes", 0)) if previous else 0
    historical_moth = SAC("MlpPolicy", moth_env, seed=args.seed + 991, device="cpu",
                          learning_rate=3e-4, buffer_size=BUFFER_SIZE,
                          batch_size=BATCH_SIZE, learning_starts=LEARNING_STARTS,
                          gamma=.99, ent_coef="auto_0.005", train_freq=1,
                          gradient_steps=1, policy_kwargs={"net_arch": [128, 128]})
    historical_moth.set_logger(configure(format_strings=[]))
    env = HuntEnv(config={"natural_environment": True, "obstacle_seed": None})
    try:
        for episode_index in range(start_episode, args.episodes):
            raw_bat, _ = env.reset(seed=args.seed + episode_index)
            bat_obs = bat_policy_observation(raw_bat)
            moth_obs = legacy_moth_observation(env.observe("moth"))
            opponent_mode, opponent_snapshot = choose_league_opponent(
                league_pool, np.random.default_rng(args.seed + episode_index + 424242),
                args.historical_moth_probability)
            if opponent_snapshot is not None:
                historical_moth.policy.load_state_dict(opponent_snapshot["state_dict"])
                historical_moth.policy.set_training_mode(False)
            bat_reward = moth_reward = 0.0
            caught = False
            last_info = {}
            while True:
                bat_action, _ = bat.predict(bat_obs, deterministic=False)
                moth_actor = historical_moth if opponent_snapshot is not None else moth
                moth_action, _ = moth_actor.predict(moth_obs, deterministic=False)
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
                "moth_opponent_mode": opponent_mode,
                "historical_moth_episode": (int(opponent_snapshot["episode"])
                                             if opponent_snapshot is not None else None),
                "obstacle_count": int(last_info["obstacle_count"]),
                "bat_replay": bat.replay_buffer.size(),
                "moth_replay": moth.replay_buffer.size(),
            })
            print(f"self-play: bat step {episode_index + 1}, moth step {episode_index + 1}"
                  f", opponent {opponent_mode}", flush=True)
            if (episode_index + 1) % 10 == 0:
                recent = episodes[-10:]
                print(f"coevolution episodes {episode_index + 1}, last10 catch "
                      f"{np.mean([e['caught'] for e in recent]):.1%}, "
                      f"bat reward {np.mean([e['bat_reward'] for e in recent]):+.3f}, "
                      f"moth reward {np.mean([e['moth_reward'] for e in recent]):+.3f}", flush=True)
            checkpoint_episode = episode_index + 1
            if checkpoint_episode % args.eval_every == 0:
                evaluation = evaluate_current_pair(
                    bat, moth, seeds=args.eval_seeds, seed_start=args.eval_seed_start)
                evaluation["episode"] = checkpoint_episode
                evaluations.append(evaluation)
                if evaluation["catch_rate"] > args.stop_threshold:
                    consecutive_passes += 1
                else:
                    consecutive_passes = 0
                print(f"league evaluation episode {checkpoint_episode}: "
                      f"{evaluation['catch_rate']:.1%} "
                      f"({evaluation['catch_count']}/{evaluation['seeds']}), "
                      f"consecutive threshold passes {consecutive_passes}", flush=True)
                league_pool = append_league_snapshot(
                    league_pool, moth, checkpoint_episode, args.league_pool_size)
            if checkpoint_episode % args.save_every == 0 or (
                    evaluations and evaluations[-1]["episode"] == checkpoint_episode):
                payload = {
                    "algorithm": "stable_baselines3_sac_league_coevolution",
                    "physics_version": PHYSICS_VERSION, "seed": args.seed,
                    "environment_config": env.cfg, "episodes": episodes,
                    "evaluations": evaluations, "updates": updates,
                    "transition_steps": transition_steps,
                    "bat_checkpoint": str(args.bat_checkpoint) if args.bat_checkpoint else None,
                    "update_every": args.update_every,
                    "moth_stationary_warmup": args.moth_stationary_warmup,
                    "league_pool_size": args.league_pool_size,
                    "historical_moth_probability": args.historical_moth_probability,
                    "eval_every": args.eval_every,
                    "eval_seed_start": args.eval_seed_start,
                    "eval_seeds": args.eval_seeds,
                    "stop_threshold": args.stop_threshold,
                    "stop_patience": args.stop_patience,
                    "consecutive_league_passes": consecutive_passes,
                }
                _save_pair(bat, moth, args.output, "running", payload, league_pool)
            if consecutive_passes >= args.stop_patience:
                print(f"league stopping after {checkpoint_episode} episodes: "
                      f"{consecutive_passes} consecutive evaluations exceeded "
                      f"{args.stop_threshold:.1%}", flush=True)
                break
    except KeyboardInterrupt:
        status = "stopped"
    else:
        status = "complete"
    payload = {
        "algorithm": "stable_baselines3_sac_league_coevolution",
        "physics_version": PHYSICS_VERSION, "seed": args.seed,
        "environment_config": env.cfg, "episodes": episodes, "evaluations": evaluations,
        "updates": updates, "transition_steps": transition_steps,
        "bat_checkpoint": str(args.bat_checkpoint) if args.bat_checkpoint else None,
        "update_every": args.update_every,
        "moth_stationary_warmup": args.moth_stationary_warmup,
        "league_pool_size": args.league_pool_size,
        "historical_moth_probability": args.historical_moth_probability,
        "eval_every": args.eval_every,
        "eval_seed_start": args.eval_seed_start,
        "eval_seeds": args.eval_seeds,
        "stop_threshold": args.stop_threshold,
        "stop_patience": args.stop_patience,
        "consecutive_league_passes": consecutive_passes,
    }
    _save_pair(bat, moth, args.output, status, payload, league_pool)
    print(f"coevolution {status}: {len(episodes)} episodes, "
          f"catch rate {np.mean([e['caught'] for e in episodes]):.1%}", flush=True)


if __name__ == "__main__":
    main()
