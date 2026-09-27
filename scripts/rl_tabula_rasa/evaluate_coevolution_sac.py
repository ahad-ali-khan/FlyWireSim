"""Held-out evaluation for a saved simultaneous SAC bat/moth pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from .env import HuntEnv, PHYSICS_VERSION
from .train_cognitive_sac import bat_policy_observation, legacy_moth_observation
from .training_runtime import atomic_json


def _run_pair(bat, moth, seed: int, mode: str):
    env = HuntEnv(config={
        "natural_environment": True,
        "obstacle_seed": None,
        "stationary_moth": mode == "stationary",
    })
    raw, _ = env.reset(seed=seed)
    bat_obs = bat_policy_observation(raw)
    moth_obs = legacy_moth_observation(env.observe("moth"))
    while True:
        bat_action, _ = bat.predict(bat_obs, deterministic=True)
        if mode == "stationary":
            moth_action = np.zeros(3, dtype=np.float32)
            moth_action[2] = -1.0
        else:
            moth_action, _ = moth.predict(moth_obs, deterministic=True)
        next_raw, _, done, _, info = env.step_with_actions({
            "bat": np.asarray(bat_action, dtype=np.float32),
            "moth": np.asarray(moth_action, dtype=np.float32),
        })
        if done:
            return bool(info["caught"]), int(info["hunt_length"])
        bat_obs = bat_policy_observation(next_raw)
        moth_obs = legacy_moth_observation(env.observe("moth"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=60000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run = json.loads(args.run.read_text())
    bat_env = HuntEnv(role="bat", config={"natural_environment": True, "obstacle_seed": None})
    moth_env = HuntEnv(role="moth", config={"natural_environment": True, "obstacle_seed": None})
    bat = SAC.load(args.run.with_suffix(".bat.zip"), env=bat_env, device="cpu")
    moth = SAC.load(args.run.with_suffix(".moth.zip"), env=moth_env, device="cpu")
    results = []
    for mode in ("learned", "stationary"):
        catches, lengths = [], []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            caught, length = _run_pair(bat, moth, seed, mode)
            catches.append(caught)
            lengths.append(length)
        result = {
            "mode": mode,
            "seeds": args.seeds,
            "seed_start": args.seed_start,
            "catch_rate": float(np.mean(catches)),
            "catch_count": int(np.sum(catches)),
            "mean_hunt_length": float(np.mean(lengths)),
        }
        results.append(result)
        print(f"coevolved pair / {mode}: {result['catch_rate']:.1%} "
              f"({result['catch_count']}/{args.seeds})", flush=True)
    atomic_json(args.output, {
        "algorithm": "stable_baselines3_sac_true_simultaneous_coevolution",
        "physics_version": PHYSICS_VERSION,
        "run": str(args.run),
        "results": results,
    })


if __name__ == "__main__":
    main()
