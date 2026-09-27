"""Evaluate matched bat checkpoints against biological and pure-RL moths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from .biological_escape import FlyWireEscapeCircuit
from .env import HuntEnv, PHYSICS_VERSION
from .train_cognitive_sac import bat_policy_observation, legacy_moth_observation
from .training_runtime import atomic_json


def evaluate_arm(bat, mode, moth_checkpoint, seeds, seed_start):
    moth = None
    if mode == "pure_rl":
        moth_env = HuntEnv(role="moth", config={"natural_environment": True, "obstacle_seed": None})
        moth = SAC.load(moth_checkpoint, env=moth_env, device="cpu")
    outcomes = []
    for seed in range(seed_start, seed_start + seeds):
        env = HuntEnv(config={"natural_environment": True, "obstacle_seed": None})
        raw, _ = env.reset(seed=seed)
        circuit = FlyWireEscapeCircuit(seed=seed) if mode == "biological" else None
        brain_spikes = 0
        looming = []
        while True:
            bat_action, _ = bat.predict(bat_policy_observation(raw), deterministic=True)
            if circuit is not None:
                moth_action, state = circuit.action(env)
                brain_spikes += int(sum(state["giant_fiber_spike_count"]))
                looming.append(float(state["looming"]))
            else:
                moth_obs = legacy_moth_observation(env.observe("moth"))
                moth_action, _ = moth.predict(moth_obs, deterministic=True)
            raw, _, done, _, info = env.step_with_actions({
                "bat": np.asarray(bat_action, dtype=np.float32),
                "moth": np.asarray(moth_action, dtype=np.float32),
            })
            if done:
                outcomes.append({
                    "seed": seed, "caught": bool(info["caught"]),
                    "hunt_length": int(info["hunt_length"]),
                    "giant_fiber_spikes": brain_spikes,
                    "mean_looming": float(np.mean(looming)) if looming else None,
                })
                break
    return {
        "moth_controller": mode,
        "seeds": seeds,
        "seed_start": seed_start,
        "catch_rate": float(np.mean([x["caught"] for x in outcomes])),
        "catch_count": int(sum(x["caught"] for x in outcomes)),
        "mean_hunt_length": float(np.mean([x["hunt_length"] for x in outcomes])),
        "mean_giant_fiber_spikes": (float(np.mean([x["giant_fiber_spikes"] for x in outcomes]))
                                     if mode == "biological" else None),
        "mean_looming": (float(np.mean([x["mean_looming"] for x in outcomes]))
                         if mode == "biological" else None),
        "episodes": outcomes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biological-bat", type=Path, required=True)
    parser.add_argument("--pure-rl-bat", type=Path, required=True)
    parser.add_argument("--pure-rl-moth", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=60_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    bat_env = HuntEnv(role="bat", config={"natural_environment": True, "obstacle_seed": None})
    biological_bat = SAC.load(args.biological_bat, env=bat_env, device="cpu")
    pure_rl_bat = SAC.load(args.pure_rl_bat, env=bat_env, device="cpu")
    results = [
        evaluate_arm(biological_bat, "biological", None, args.seeds, args.seed_start),
        evaluate_arm(pure_rl_bat, "pure_rl", args.pure_rl_moth, args.seeds, args.seed_start),
    ]
    for result in results:
        print(f"{result['moth_controller']}: {result['catch_rate']:.1%} "
              f"({result['catch_count']}/{result['seeds']})", flush=True)
    atomic_json(args.output, {
        "algorithm": "matched_bat_training_biological_vs_pure_rl_moth",
        "physics_version": PHYSICS_VERSION,
        "biological_circuit": {
            "pathway": "LPLC2 -> DNp01/Giant Fibre -> motor readout",
            "lplc2_root_id": 720575940640302389,
            "giant_fiber_root_id": 720575940622838154,
            "connectome_synapses": 3,
        },
        "results": results,
    })


if __name__ == "__main__":
    main()
