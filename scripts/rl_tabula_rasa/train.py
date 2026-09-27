"""Run Phase 0: continuous-action PPO and an auditable learning-curve gate.

Usage: uv run python -m scripts.rl_tabula_rasa.train --steps 100000
The bat is first trained against a stochastic continuous-control moth; this is
an opponent bootstrap, not a claim of simultaneous coevolution. If the bat gate
passes, the moth gets its own PPO training against that frozen bat policy.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .connectome import apply_seed, load_seed
from .env import CONFIG, HuntEnv, assert_3d_motion


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
LOG_INTERVAL = 1000
MOVING_WINDOW = 100
MEANINGFUL_MARGIN = 0.10
DEFAULT_STEPS = 100_000
PPO_CONFIG = {"n_steps": 1024, "batch_size": 256, "n_epochs": 5,
              "learning_rate": 3e-4, "entropy_coefficient": 0.005}


def moving_rates(catches: list[int], steps: list[int]) -> list[dict]:
    window: deque[int] = deque(maxlen=MOVING_WINDOW)
    points = []
    for episode, (catch, step) in enumerate(zip(catches, steps), 1):
        window.append(catch)
        points.append({"episode": episode, "step": step,
                       "catch_rate_last_100": float(np.mean(window))})
    return points


def gate(points: list[dict]) -> dict:
    if len(points) < 20:
        return {"passed": False, "reason": "fewer than 20 completed hunts"}
    values = np.asarray([p["catch_rate_last_100"] for p in points], dtype=float)
    half = len(values) // 2
    tail = values[half:]
    slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0])
    decile = max(1, len(values) // 10)
    first = float(values[:decile].mean())
    last = float(values[-decile:].mean())
    improvement = last - first
    return {"passed": bool(slope > 0 and improvement >= MEANINGFUL_MARGIN),
            "second_half_slope_per_episode": slope,
            "first_decile_average": first, "final_decile_average": last,
            "absolute_improvement": improvement,
            "required_absolute_improvement": MEANINGFUL_MARGIN}


class CurveCallback(BaseCallback):
    def __init__(self, role: str, total_steps: int, seed: int,
                 catches: list[int] | None = None, episode_steps: list[int] | None = None,
                 artifact_prefix: str | None = None, connectome_metadata: dict | None = None):
        super().__init__()
        self.role, self.total_steps, self.seed = role, total_steps, seed
        self.artifact_prefix = artifact_prefix or ("phase0" if role == "bat" else "phase0_moth")
        self.connectome_metadata = connectome_metadata
        self.catches = list(catches or [])
        self.episode_steps = list(episode_steps or [])
        self.next_log = LOG_INTERVAL

    def _on_training_start(self):
        self.next_log = ((self.num_timesteps // LOG_INTERVAL) + 1) * LOG_INTERVAL

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.catches.append(int(info["caught"]))
                self.episode_steps.append(self.num_timesteps)
        if self.num_timesteps >= self.next_log:
            self.save()
            recent = np.mean(self.catches[-MOVING_WINDOW:]) if self.catches else 0.0
            metric = "bat catch"
            if self.role == "moth":
                recent = 1.0 - recent
                metric = "moth survival"
            print(f"{self.role}: step {self.num_timesteps}, hunts {len(self.catches)}, "
                  f"recent {metric} rate {recent:.3f}",
                  flush=True)
            self.next_log += LOG_INTERVAL
        return True

    def save(self):
        RESULTS.mkdir(exist_ok=True)
        points = moving_rates(self.catches, self.episode_steps)
        if self.role == "moth":
            for point in points:
                point["moth_survival_rate_last_100"] = 1.0 - point["catch_rate_last_100"]
        phase_number = int(self.artifact_prefix.removeprefix("phase").split("_", 1)[0])
        result = {"phase": phase_number, "role": self.role, "seed": self.seed,
                  "collision_model": "jaw offset measured from Blender Armature_Bat Jaw_Lower bone; animated jaw-distance tolerance",
                  "training_steps_completed": self.num_timesteps,
                  "target_training_steps": self.total_steps,
                  "opponent": "random continuous actions" if self.role == "bat" else "frozen trained bat PPO",
                  "algorithm": "PPO", "reward_config": CONFIG,
                  "ppo_config": PPO_CONFIG,
                  "gate": gate(points) if self.role == "bat" else None,
                  "curve": points,
                  "episode_steps": self.episode_steps,
                  "episode_catches": self.catches}
        result["connectome_seed"] = self.connectome_metadata
        path = RESULTS / f"{self.artifact_prefix}_learning_curve.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        metric_key = "catch_rate_last_100" if self.role == "bat" else "moth_survival_rate_last_100"
        metric_label = "Bat catch rate" if self.role == "bat" else "Moth survival rate"
        ax.plot([p["step"] for p in points],
                [p[metric_key] for p in points], color="#18a5ca", linewidth=2)
        phase_label = self.artifact_prefix.split("_", 1)[0].upper()
        ax.set(title=f"Phase {phase_label.removeprefix('PHASE')} — {metric_label}, last {MOVING_WINDOW} hunts",
               xlabel="Training steps", ylabel=metric_label, ylim=(0, 1))
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(RESULTS / f"{self.artifact_prefix}_curve.png", dpi=140)
        plt.close(fig)


def train(role: str, steps: int, seed: int, opponent=None,
          resume: bool = False, connectome_seed: tuple[np.ndarray, dict] | None = None,
          artifact_prefix: str | None = None) -> tuple[PPO, CurveCallback]:
    env = Monitor(HuntEnv(role=role, opponent=opponent))
    env.reset(seed=seed)
    artifact_prefix = artifact_prefix or ("phase0" if role == "bat" else "phase0_moth")
    checkpoint_stem = ("phase0_bat" if role == "bat" and artifact_prefix == "phase0"
                       else artifact_prefix)
    checkpoint = RESULTS / f"{checkpoint_stem}_ppo.zip"
    resume = resume and checkpoint.exists()
    if resume:
        model = PPO.load(checkpoint, env=env, device="cpu")
        prior_path = RESULTS / f"{artifact_prefix}_learning_curve.json"
        prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
        catches = prior.get("episode_catches", [])
        episode_steps = [int(point["step"]) for point in prior.get("curve", [])]
        # The saved curve contains one point per episode, so retain exact episode
        # step indices rather than reconstructing them from the moving averages.
        episode_steps = prior.get("episode_steps", episode_steps)
        target_steps = int(prior.get("training_steps_completed", model.num_timesteps)) + steps
        callback = CurveCallback(role, target_steps, seed, catches, episode_steps,
                                 artifact_prefix, prior.get("connectome_seed"))
    else:
        model = PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu",
                    n_steps=PPO_CONFIG["n_steps"], batch_size=PPO_CONFIG["batch_size"],
                    n_epochs=PPO_CONFIG["n_epochs"], learning_rate=PPO_CONFIG["learning_rate"],
                    ent_coef=PPO_CONFIG["entropy_coefficient"])
        callback = CurveCallback(role, steps, seed, artifact_prefix=artifact_prefix,
                                 connectome_metadata=connectome_seed[1] if connectome_seed else None)
        if connectome_seed is not None:
            if role != "moth":
                raise ValueError("FlyWire optic-pathway seeding applies to the moth only")
            apply_seed(model, connectome_seed[0])
    model.learn(total_timesteps=steps, callback=callback, reset_num_timesteps=not resume)
    callback.save()
    model.save(checkpoint.with_suffix(""))
    env.close()
    return model, callback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--moth-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--resume", action="store_true",
                        help="continue Phase 0 weights and append to their saved learning curve")
    parser.add_argument("--skip-bat", action="store_true",
                        help="reuse the gated Phase 0 bat checkpoint and train only the moth")
    parser.add_argument("--moth-connectome-seed", type=Path,
                        help="seed the moth's visual input weights from a built pathway matrix")
    args = parser.parse_args()
    if args.steps < 1000 or args.moth_steps < 1000:
        parser.error("steps and moth-steps must be at least 1000")
    assert_3d_motion()  # startup regression check
    gate_path = RESULTS / "phase0_learning_curve.json"
    if args.skip_bat:
        if not gate_path.exists() or not json.loads(gate_path.read_text()).get("gate", {}).get("passed"):
            parser.error("--skip-bat requires a passing saved Phase 0 bat gate")
        bat = PPO.load(RESULTS / "phase0_bat_ppo.zip", device="cpu")
    else:
        bat, progress = train("bat", args.steps, args.seed, resume=args.resume)
        decision = gate(moving_rates(progress.catches, progress.episode_steps))
        print("Phase 0 bat learning gate:", decision, flush=True)
        if not decision["passed"]:
            print("STOP: learning gate did not pass; no later phases or moth training started.", flush=True)
            return 1
    connectome_seed = load_seed(args.moth_connectome_seed) if args.moth_connectome_seed else None
    output_prefix = "phase3_moth_connectome_seeded" if connectome_seed else "phase0_moth"
    train("moth", args.moth_steps, args.seed + 1, opponent=bat, resume=args.resume,
          connectome_seed=connectome_seed, artifact_prefix=output_prefix)
    print(f"PPO models and learning curves saved under results/ ({output_prefix}).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
