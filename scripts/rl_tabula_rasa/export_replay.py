"""Record real PPO rollouts for the Blender viewer, without altering training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from .env import CONFIG, HuntEnv


def snapshot(env: HuntEnv, episode: int, outcome: str = "hunting", info=None) -> dict:
    mouth = env.mouth_position()
    distance = float(((mouth - env.moth.position) ** 2).sum() ** 0.5)
    frame = {"episode": episode, "tick": env.tick,
            "bat_position": env.bat.position.tolist(), "bat_velocity": env.bat.velocity.tolist(),
            "bat_mouth_position": mouth.tolist(),
            "moth_position": env.moth.position.tolist(), "moth_velocity": env.moth.velocity.tolist(),
            "flame_position": env.flame.tolist(), "distance": distance,
            "bat_sensor_hit": distance <= CONFIG["sonar_range"],
            "outcome": outcome}
    if info is not None:
        frame["jaw_contact_fraction"] = info["jaw_contact_fraction"]
        frame["jaw_contact_position"] = info["jaw_contact_position"]
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bat-model", default="results/phase0_bat_ppo.zip")
    parser.add_argument("--moth-model", default="results/phase0_moth_ppo.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42_000)
    parser.add_argument("--output-file", default="results/phase0_replay.json")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    bat = PPO.load(args.bat_model, device="cpu")
    moth = PPO.load(args.moth_model, device="cpu")
    env = HuntEnv("bat", opponent=moth)
    frames = []
    outcomes = []
    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + episode)
        frames.append(snapshot(env, episode))
        done = False
        while not done:
            action = bat.predict(obs, deterministic=True)[0]
            obs, _, done, _, info = env.step(action)
            frames.append(snapshot(env, episode,
                                   "caught_model_contact" if info["caught"] else
                                   "survived" if info["timeout"] else "hunting", info))
        outcomes.append({"episode": episode, "caught": bool(info["caught"]),
                         "length": env.tick})
    connectome_seeded = "connectome_seeded" in args.moth_model
    phase = (3 if connectome_seeded else
             2 if "phase2" in args.bat_model and "phase2" in args.moth_model else 0)
    report = {"schema": 1, "source": f"Phase {phase} PPO rollout, not Blender physics",
              "policy_phase": phase,
              "connectome_seeded_moth": connectome_seeded,
              "bat_model": args.bat_model, "moth_model": args.moth_model,
              "seed_start": args.seed, "episodes": outcomes, "frames": frames,
              "collision_note": "catch labels use synchronized swept jaw/moth collision in the trainer; jaw-contact markers use the recorded collision point and tick fraction"}
    Path(args.output_file).write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(frames)} real PPO frames to {args.output_file}; "
          f"{sum(o['caught'] for o in outcomes)}/{args.episodes} model catches")


if __name__ == "__main__":
    main()
