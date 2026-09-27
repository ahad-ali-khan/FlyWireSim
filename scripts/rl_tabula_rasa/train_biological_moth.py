"""Train a bat against either the connectome-grounded or pure-RL moth.

The bat learner, physics, spawn process, and training budget are identical
between arms.  Only the fixed moth controller changes, making this a matched
comparison rather than a claim that the small escape circuit is a whole-brain
simulation.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from .biological_escape import FlyWireEscapeCircuit
from .env import HuntEnv, PHYSICS_VERSION
from .train_cognitive_sac import bat_policy_observation, legacy_moth_observation
from .training_runtime import atomic_json


class FixedMothEnv(HuntEnv):
    """Bat-facing SB3 environment with one explicitly selected moth arm."""

    def __init__(self, mode: str, seed: int, moth_checkpoint: Path | None = None):
        super().__init__(role="bat", config={"natural_environment": True, "obstacle_seed": None})
        self.mode = mode
        self.run_seed = seed
        self.moth_checkpoint = str(moth_checkpoint) if moth_checkpoint else None
        self.moth_policy = None
        self.circuit = FlyWireEscapeCircuit(seed=seed) if mode == "biological" else None

    def reset(self, **kwargs):
        raw, info = super().reset(**kwargs)
        if self.circuit is not None:
            self.circuit.reset()
        return bat_policy_observation(raw), info

    def step(self, action):
        brain = None
        if self.mode == "biological":
            moth_action, brain = self.circuit.action(self)
        else:
            moth_obs = legacy_moth_observation(self.observe("moth"))
            moth_action, _ = self.moth_policy.predict(moth_obs, deterministic=True)
            moth_action = np.asarray(moth_action, dtype=np.float32)
        raw, reward, done, truncated, info = self.step_with_actions({
            "bat": np.asarray(action, dtype=np.float32),
            "moth": moth_action,
        })
        if brain is not None:
            info["moth_brain"] = brain
        info["moth_controller"] = self.mode
        return bat_policy_observation(raw), reward, done, truncated, info


class MatchedRunCallback(BaseCallback):
    def __init__(self, output: Path, mode: str, config: dict, seed: int):
        super().__init__()
        self.output, self.mode, self.config, self.seed = output, mode, config, seed
        self.episodes = []
        self.reward = 0.0
        self.looming = []
        self.gf_spikes = 0
        self.lplc2_spikes = 0

    def _manifest(self, status: str) -> dict:
        return {
            "algorithm": "stable_baselines3_sac_bat_fixed_moth_comparison",
            "physics_version": PHYSICS_VERSION,
            "seed": self.seed,
            "moth_controller": self.mode,
            "environment_config": self.config,
            "total_steps": self.num_timesteps,
            "status": status,
            "episodes": self.episodes,
        }

    def _save(self, status: str):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.output.parent, prefix=".biomoth-save-") as tmp:
            model_tmp = Path(tmp) / "model.zip"
            replay_tmp = Path(tmp) / "replay.pkl"
            self.model.save(model_tmp)
            self.model.save_replay_buffer(replay_tmp)
            os.replace(model_tmp, self.output.with_suffix(".zip"))
            os.replace(replay_tmp, self.output.with_suffix(".replay.pkl"))
        atomic_json(self.output, self._manifest(status))

    def _on_training_start(self):
        self.model.set_logger(configure(format_strings=[]))

    def _on_step(self):
        self.reward += float(self.locals["rewards"][0])
        info = self.locals["infos"][0]
        brain = info.get("moth_brain")
        if brain is not None:
            self.looming.append(float(brain["looming"]))
            self.gf_spikes += int(sum(brain["giant_fiber_spike_count"]))
            self.lplc2_spikes += int(sum(brain["lplc2_spike_count"]))
        if self.locals["dones"][0]:
            self.episodes.append({
                "episode": len(self.episodes) + 1,
                "environment_steps": self.num_timesteps,
                "caught": bool(info["caught"]),
                "reward": self.reward,
                "hunt_length": int(info["hunt_length"]),
                "mean_looming": float(np.mean(self.looming)) if self.looming else 0.0,
                "giant_fiber_spikes": self.gf_spikes,
                "lplc2_spikes": self.lplc2_spikes,
            })
            self.reward, self.looming, self.gf_spikes, self.lplc2_spikes = 0.0, [], 0, 0
            if len(self.episodes) % 25 == 0:
                recent = self.episodes[-25:]
                print(f"{self.mode}: episode {len(self.episodes)}, last25 catch "
                      f"{np.mean([e['caught'] for e in recent]):.1%}, "
                      f"reward {np.mean([e['reward'] for e in recent]):+.3f}", flush=True)
                self._save("running")
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--moth-mode", choices=("biological", "pure_rl"), required=True)
    parser.add_argument("--moth-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.moth_mode == "pure_rl" and args.moth_checkpoint is None:
        parser.error("--moth-checkpoint is required for --moth-mode pure_rl")
    if any(args.output.with_suffix(s).exists() for s in (".json", ".zip", ".replay.pkl")):
        parser.error("refusing to overwrite an existing matched comparison")

    torch.set_num_threads(1)
    env = FixedMothEnv(args.moth_mode, args.seed, args.moth_checkpoint)
    if args.moth_mode == "pure_rl":
        moth_env = HuntEnv(role="moth", config={"natural_environment": True, "obstacle_seed": None})
        env.moth_policy = SAC.load(args.moth_checkpoint, env=moth_env, device="cpu")
    model = SAC("MlpPolicy", env, seed=args.seed, device="cpu", learning_rate=3e-4,
                buffer_size=20_000, batch_size=128, learning_starts=1_000,
                gamma=.99, ent_coef="auto_0.005", train_freq=1, gradient_steps=1,
                policy_kwargs={"net_arch": [128, 128]})
    callback = MatchedRunCallback(args.output, args.moth_mode, env.cfg.copy(), args.seed)
    try:
        model.learn(args.steps, callback=callback, reset_num_timesteps=True)
    except KeyboardInterrupt:
        callback._save("stopped")
        print(f"{args.moth_mode}: stopped after {model.num_timesteps} steps", flush=True)
    else:
        callback._save("complete")
        print(f"{args.moth_mode}: complete, {len(callback.episodes)} episodes", flush=True)


if __name__ == "__main__":
    main()
