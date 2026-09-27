"""Stable-Baselines3 SAC control experiment; NOT the cognitive architecture.

Uses installed Stable-Baselines3 to isolate custom recurrent learner failures.
No demonstrations, pursuit labels, scripted steering, or reward overrides.
"""
import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from .cognitive import CognitiveActorCritic
from .env import HuntEnv, PHYSICS_VERSION
from .evaluate_cognitive_sac import evaluate
from .train_cognitive_sac import bat_policy_observation, legacy_moth_observation
from .training_runtime import atomic_json


class MixedOpponentEnv(HuntEnv):
    def __init__(self, moth, seed, curriculum_steps=0):
        super().__init__(config={"natural_environment": True, "obstacle_seed": seed})
        self.moth_policy = moth
        self.total_steps = 0
        self.curriculum_steps = curriculum_steps

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.opponent_mode = self.np_random.choice(["learned", "random", "stationary"], p=[.5, .3, .2])
        if self.total_steps < self.curriculum_steps:
            progress = self.total_steps / self.curriculum_steps
            difficulty = float(np.clip((progress - .25) / .75, 0, 1))
            if self.np_random.random() > difficulty:
                self.opponent_mode = "stationary"
            delta = self.moth.position - self.bat.position
            spread = .3 + (math.pi - .3) * difficulty
            self.bat.yaw = math.atan2(delta[1], delta[0]) + self.np_random.uniform(-spread, spread)
            self.bat.pitch = float(np.clip(math.atan2(delta[2], np.hypot(delta[0], delta[1])) +
                self.np_random.uniform(-.1 - .25 * difficulty, .1 + .25 * difficulty),
                self.cfg["min_pitch"], self.cfg["max_pitch"]))
        self.cfg["stationary_moth"] = self.opponent_mode == "stationary"
        self.hidden = self.moth_policy.initial_hidden()
        return bat_policy_observation(self.observe("bat")), info

    def step(self, action):
        self.total_steps += 1
        if self.opponent_mode == "learned":
            obs = torch.tensor(legacy_moth_observation(self.observe("moth"))).reshape(1, 1, -1)
            with torch.no_grad():
                output = self.moth_policy.forward_sequence(obs, self.hidden)
                self.hidden = output.hidden
                other = torch.distributions.Normal(output.mean[0, 0], self.moth_policy.log_standard_deviation.exp()).sample().clamp(-1, 1).numpy()
        elif self.opponent_mode == "random":
            other = self.np_random.uniform(-1, 1, 3).astype(np.float32)
        else:
            other = np.array([0, 0, -1], dtype=np.float32)
        obs, reward, done, truncated, info = self.step_with_actions({"bat": action, "moth": other})
        info["moth_opponent_mode"] = str(self.opponent_mode)
        return bat_policy_observation(obs), reward, done, truncated, info


class FlatEvaluationAdapter:
    """Let the shared evaluator compare a library policy on identical hunts."""
    def __init__(self, model):
        self.model = model

    def initial_hidden(self):
        return None

    def forward_sequence(self, observation, hidden=None):
        action, _ = self.model.predict(observation[0, 0].numpy(), deterministic=True)
        raw = torch.tensor(np.arctanh(np.clip(action, -.999999, .999999))).reshape(1, 1, 3)
        return SimpleNamespace(mean=raw, hidden=None)


class AuditCallback(BaseCallback):
    def __init__(self, moth, output, config):
        super().__init__()
        self.moth, self.output, self.config = moth, output, config
        self.episodes, self.evaluations = [], []
        self.reward = 0.

    def write(self, status):
        payload = dict(algorithm="stable_baselines3_sac_flat_control",
            cognitive_architecture=False, physics_version=PHYSICS_VERSION,
            seed=self.model.seed, environment_config=self.config, observation_frame="body",
            curriculum_steps=self.training_env.get_attr("curriculum_steps")[0],
            total_steps=self.num_timesteps, status=status,
            resume_boundary="new episode; policy, critics, optimizers, entropy and replay retained",
            episodes=self.episodes, evaluations=self.evaluations)
        atomic_json(self.output, payload)
        return payload

    def save_checkpoint(self, status):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.output.parent, prefix=".sac-save-") as tmp:
            model_path, replay_path = Path(tmp) / "model.zip", Path(tmp) / "replay.pkl"
            self.model.save(model_path)
            self.model.save_replay_buffer(replay_path)
            os.replace(model_path, self.output.with_suffix(".zip"))
            os.replace(replay_path, self.output.with_suffix(".replay.pkl"))
        atomic_json(self.output.with_suffix(".checkpoint.json"), self.write(status))

    def evaluate_now(self):
        result = evaluate(FlatEvaluationAdapter(self.model), self.moth, self.config,
            seeds=50, observation_frame="body")
        self.evaluations.append(dict(steps=self.num_timesteps, gradient_updates=self.model._n_updates, **result))
        print(f"FLAT SAC heldout at {self.num_timesteps}: {result['catch_rate']:.1%}", flush=True)

    def _on_training_start(self):
        self.evaluate_now()

    def _on_step(self):
        self.reward += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            info = self.locals["infos"][0]
            self.episodes.append(dict(episode=len(self.episodes)+1, caught=info["caught"],
                environment_steps=self.num_timesteps,
                gradient_updates_before_episode_end=self.model._n_updates,
                reward=self.reward, moth_opponent_mode=info["moth_opponent_mode"]))
            self.reward = 0.
            if len(self.episodes) % 50 == 0:
                print(f"FLAT SAC episodes {len(self.episodes)}, last50 "
                      f"catch {np.mean([e['caught'] for e in self.episodes[-50:]]):.1%}", flush=True)
                self.write("running")
        if self.num_timesteps % 10000 == 0:
            self.evaluate_now()
            self.save_checkpoint("running")
            self.model.save(self.output.with_name(f"{self.output.stem}_step{self.num_timesteps:06d}.zip"))
            self.write("running")
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--curriculum-steps", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="continue to --steps total transitions")
    parser.add_argument("--moth-opponent-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.resume and (args.output.exists() or args.output.with_suffix(".zip").exists()):
        parser.error("refusing to overwrite an existing control experiment")
    if args.resume and not all(args.output.with_suffix(s).is_file() for s in (".json", ".zip", ".replay.pkl")):
        parser.error("resume requires run JSON, model .zip and replay .replay.pkl")
    torch.set_num_threads(1)
    moth = CognitiveActorCritic("moth", 23).eval()
    moth.load_state_dict(torch.load(args.moth_opponent_checkpoint, map_location="cpu", weights_only=False)["models"]["moth"])
    env = MixedOpponentEnv(moth, args.seed, args.curriculum_steps)
    config = env.cfg.copy()
    previous = None
    if args.resume:
        manifest = args.output.with_suffix(".checkpoint.json")
        previous = json.loads((manifest if manifest.exists() else args.output).read_text())
        if (previous["seed"] != args.seed or previous["environment_config"] != config or
                previous["physics_version"] != PHYSICS_VERSION or
                previous.get("curriculum_steps", 0) != args.curriculum_steps):
            parser.error("resume configuration differs from the saved experiment")
        model = SAC.load(args.output.with_suffix(".zip"), env=env, device="cpu")
        model.load_replay_buffer(args.output.with_suffix(".replay.pkl"))
        env.total_steps = model.num_timesteps
        env.reset(seed=args.seed + model.num_timesteps)
        print(f"Resuming at {model.num_timesteps} steps with {model.replay_buffer.size()} replay transitions", flush=True)
    else:
        model = SAC("MlpPolicy", env, seed=args.seed, device="cpu", learning_rate=3e-4,
            buffer_size=20000, batch_size=128, learning_starts=1000, gamma=.99,
            ent_coef="auto_0.005", train_freq=1, gradient_steps=1,
            policy_kwargs={"net_arch": [128, 128]})
    if args.steps <= model.num_timesteps:
        parser.error("--steps must exceed the already completed transition count")
    callback = AuditCallback(moth, args.output, config)
    if previous is not None:
        callback.episodes = previous["episodes"]
        callback.evaluations = previous["evaluations"]
    try:
        model.learn(args.steps - model.num_timesteps, callback=callback, reset_num_timesteps=not args.resume)
    except KeyboardInterrupt:
        status = "stopped"
    else:
        callback.evaluate_now()
        status = "complete"
    callback.save_checkpoint(status)


if __name__ == "__main__":
    main()
