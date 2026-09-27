"""Evaluate the deterministic pursuit baseline with swept jaw contact."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .baselines import deterministic_pursuit_baseline, random_action
from .env import CONFIG, PHYSICS_VERSION, HuntEnv


def evaluate(seeds: int, seed_start: int) -> dict:
    outcomes = []
    for seed in range(seed_start, seed_start + seeds):
        env = HuntEnv("bat")
        env.reset(seed=seed)
        rng = np.random.default_rng(seed + 1)
        info = {"caught": False}
        for _ in range(env.cfg["max_ticks"]):
            bat_action = deterministic_pursuit_baseline(
                env.bat.position, env.moth.position, env.bat.velocity,
                env.bat.yaw, env.bat.pitch,
                max_yaw_delta=env.cfg["max_yaw_delta"],
                max_pitch_delta=env.cfg["max_pitch_delta"],
                bat_accel=env.cfg["bat_accel"], drag=env.cfg["drag"],
                dt=env.cfg["dt"],
                target_speed=env.cfg["bat_max_speed"])
            _, _, done, _, info = env.step_with_actions({
                "bat": bat_action, "moth": random_action(rng)})
            if done:
                break
        endpoint_catch = (info["jaw_distance"] <= env.cfg["jaw_contact_radius"])
        outcomes.append({"seed": seed, "caught_swept": bool(info["caught"]),
                         "caught_endpoint": bool(endpoint_catch),
                         "hunt_length": int(info["hunt_length"])})

    swept = np.asarray([row["caught_swept"] for row in outcomes], dtype=float)
    endpoint = np.asarray([row["caught_endpoint"] for row in outcomes], dtype=float)
    missed_by_endpoint = sum(row["caught_swept"] and not row["caught_endpoint"]
                             for row in outcomes)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": "deterministic_pursuit_baseline",
        "opponent": "random_action_baseline",
        "collision": "synchronous relative-motion sweep; earliest radius contact",
        "physics_version": PHYSICS_VERSION,
        "environment_config": CONFIG.copy(),
        "seed_start": seed_start,
        "seed_count": seeds,
        "jaw_contact_radius": CONFIG["jaw_contact_radius"],
        "catch_rate_swept": float(swept.mean()) if seeds else None,
        "catch_rate_endpoint_only": float(endpoint.mean()) if seeds else None,
        "swept_catches_missed_by_endpoint": int(missed_by_endpoint),
        "episodes": outcomes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=23000)
    parser.add_argument("--output", type=Path,
                        default=Path("results/pursuit_swept_dt016_500seed.json"))
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    result = evaluate(args.seeds, args.seed_start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['baseline']} swept catch rate: {result['catch_rate_swept']:.1%} "
          f"({result['catch_rate_endpoint_only']:.1%} endpoint-only); "
          f"{result['swept_catches_missed_by_endpoint']}/{args.seeds} catches were "
          f"previously invisible; results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
