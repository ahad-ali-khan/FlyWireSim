"""Held-out bat comparison against random continuous actions; no training writes."""

from __future__ import annotations

import argparse
import json

import numpy as np
from stable_baselines3 import PPO

from .env import HuntEnv
from .train import RESULTS


def evaluate(model, episodes: int, seed: int, random_actor: bool):
    env = HuntEnv("bat")
    rng = np.random.default_rng(seed + 7_000_000)
    catches = []
    lengths = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        env.action_space.seed(seed + episode)
        done = False
        while not done:
            action = (rng.uniform(-1, 1, size=3).astype(np.float32) if random_actor
                      else model.predict(obs, deterministic=True)[0])
            obs, _, done, _, info = env.step(action)
        catches.append(int(info["caught"]))
        lengths.append(info["hunt_length"])
    return {"episodes": episodes, "catch_rate": float(np.mean(catches)),
            "average_hunt_length": float(np.mean(lengths)), "catches": sum(catches)}


def evaluate_moth(model, bat, episodes: int, seed: int, random_actor: bool):
    """Compare survival against one frozen bat policy on held-out randomized hunts."""
    env = HuntEnv("moth", opponent=bat)
    rng = np.random.default_rng(seed + 8_000_000)
    escapes = []
    lengths = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        env.action_space.seed(seed + episode)
        done = False
        while not done:
            action = (rng.uniform(-1, 1, size=3).astype(np.float32) if random_actor
                      else model.predict(obs, deterministic=True)[0])
            obs, _, done, _, info = env.step(action)
        escapes.append(int(info["timeout"]))
        lengths.append(info["hunt_length"])
    return {"episodes": episodes, "survival_rate": float(np.mean(escapes)),
            "average_hunt_length": float(np.mean(lengths)), "survivals": sum(escapes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--model", default="results/phase0_bat_ppo.zip")
    parser.add_argument("--output-file", default="results/phase0_holdout.json")
    parser.add_argument("--role", choices=("bat", "moth"), default="bat")
    parser.add_argument("--opponent-model", default="results/phase0_bat_ppo.zip")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    from pathlib import Path
    model = PPO.load(Path(args.model), device="cpu")
    if args.role == "bat":
        result = {"setting": "held-out randomized starts, stochastic continuous moth",
                  "seed_start": args.seed,
                  "trained_bat": evaluate(model, args.episodes, args.seed, False),
                  "random_bat": evaluate(model, args.episodes, args.seed, True)}
    else:
        opponent = PPO.load(Path(args.opponent_model), device="cpu")
        result = {"setting": "held-out randomized starts against frozen trained bat",
                  "seed_start": args.seed,
                  "trained_moth": evaluate_moth(model, opponent, args.episodes, args.seed, False),
                  "random_moth": evaluate_moth(model, opponent, args.episodes, args.seed, True)}
    Path(args.output_file).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
