"""Run the saved deterministic SAC bat against a stationary moth."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .cognitive import CognitiveActorCritic
from .env import PHYSICS_VERSION
from .novelty import NoveltyEnv
from .train_cognitive_sac import bat_policy_observation
from .training_runtime import ROOT, atomic_json


def evaluate(checkpoint_path: Path, seeds: int, seed_start: int,
             capture_trajectories: set[int] | None = None) -> dict:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAC checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "bat" not in checkpoint:
        raise ValueError(f"checkpoint has no SAC bat policy: {checkpoint_path}")
    if checkpoint.get("physics_version") != PHYSICS_VERSION or "environment_config" not in checkpoint:
        raise ValueError("Legacy checkpoint has no compatible physics configuration. Its archived "
                         "diagnostic remains valid only for that historical environment; do not "
                         "silently evaluate it under today's dynamics.")

    torch.set_num_threads(1)
    bat = CognitiveActorCritic("bat", 19).eval()
    bat.load_state_dict(checkpoint["bat"])
    env = NoveltyEnv("bat", config=checkpoint["environment_config"] | {"stationary_moth": True})
    outcomes = []
    trajectories = {}
    capture_trajectories = capture_trajectories or set()
    for seed in range(seed_start, seed_start + seeds):
        env.reset(seed=seed)
        hidden = bat.initial_hidden().detach()
        info = {"caught": False}
        trajectory = []
        if seed in capture_trajectories:
            trajectory.append({"tick": 0,
                               "bat_position": env.bat.position.tolist(),
                               "moth_position": env.moth.position.tolist(),
                               "bat_velocity": env.bat.velocity.tolist(),
                               "jaw_distance": float(np.linalg.norm(
                                   env.mouth_position() - env.moth.position))})
        for _ in range(env.cfg["max_ticks"]):
            observation = torch.as_tensor(bat_policy_observation(env.observe("bat"),
                checkpoint.get("observation_frame", "world")), dtype=torch.float32)
            with torch.no_grad():
                output = bat.forward_sequence(observation.reshape(1, 1, -1), hidden)
                action = torch.tanh(output.mean[0, 0]).cpu().numpy()
            hidden = output.hidden.detach()
            _, _, done, _, info = env.step_joint({
                "bat": action,
                "moth": np.zeros(3, dtype=np.float32),
            })
            if seed in capture_trajectories:
                trajectory.append({"tick": int(env.tick),
                                   "action_yaw_pitch_thrust": action.tolist(),
                                   "bat_position": env.bat.position.tolist(),
                                   "moth_position": env.moth.position.tolist(),
                                   "bat_velocity": env.bat.velocity.tolist(),
                                   "jaw_distance": float(info["jaw_distance"])})
            if done:
                break
        outcomes.append({"seed": seed, "caught": bool(info["caught"]),
                         "hunt_length": int(info["hunt_length"])})
        if seed in capture_trajectories:
            trajectories[str(seed)] = {
                "caught": bool(info["caught"]),
                "hunt_length": int(info["hunt_length"]),
                "initial_bat_moth_distance": float(np.linalg.norm(
                    np.asarray(trajectory[0]["bat_position"])
                    - np.asarray(trajectory[0]["moth_position"]))),
                "ticks": trajectory,
            }

    caught = sum(row["caught"] for row in outcomes)
    catch_ticks = [row["hunt_length"] for row in outcomes if row["caught"]]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "sac-vs-stationary",
        "checkpoint": str(checkpoint_path),
        "policy_action": "tanh(actor_mean), deterministic",
        "opponent": "stationary_moth",
        "natural_environment": env.cfg["natural_environment"],
        "obstacle_seed": env.cfg["obstacle_seed"],
        "physics_version": PHYSICS_VERSION,
        "environment_config": checkpoint["environment_config"],
        "observation_frame": checkpoint.get("observation_frame", "world"),
        "seed_start": seed_start,
        "seed_count": seeds,
        "max_ticks": env.cfg["max_ticks"],
        "jaw_contact_radius": env.cfg["jaw_contact_radius"],
        "catches": caught,
        "catch_rate": caught / seeds if seeds else None,
        "mean_catch_tick": float(np.mean(catch_ticks)) if catch_ticks else None,
        "episodes": outcomes,
        "trajectories": trajectories,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sac-vs-stationary",),
                        default="sac-vs-stationary")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "data/cognitive_sac_checkpoint.pt")
    parser.add_argument("--seeds", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=23000)
    parser.add_argument("--capture-trajectories", type=int, nargs="*", default=[],
                        help="also record tick-by-tick positions/actions for these seeds")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/sac_vs_stationary_500seed.json")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    result = evaluate(args.checkpoint, args.seeds, args.seed_start,
                      set(args.capture_trajectories))
    atomic_json(args.output, result)
    print(f"SAC vs stationary moth: {result['catch_rate']:.1%} "
          f"({result['catches']}/{args.seeds}); results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
