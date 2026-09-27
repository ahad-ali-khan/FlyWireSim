"""Headless pacing, live-state publication, and resumable PPO checkpoints."""

from __future__ import annotations

import base64
import json
import os
import random
import re
import subprocess
import tempfile
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from .environment_config import TERRAIN_DISPLACEMENT_SCALE, TERRAIN_NOISE_FREQUENCY
from .train import RESULTS


ROOT = RESULTS.parent
DATA = ROOT / "data"
TICKS_PER_SECOND = float(os.environ.get("TICKS_PER_SECOND", "30"))
FULL_CHECKPOINT_INTERVAL_MINUTES = float(os.environ.get("FULL_CHECKPOINT_INTERVAL_MINUTES", "5"))
LIVE_STATE_INTERVAL_TICKS = int(os.environ.get("LIVE_STATE_INTERVAL_TICKS", "1"))
TRAINING_MAX_THREADS = int(os.environ.get("TRAINING_MAX_THREADS", "2"))
TRAINING_MIN_BATTERY_PCT = int(os.environ.get("TRAINING_MIN_BATTERY_PCT", "20"))
if TICKS_PER_SECOND <= 0 or FULL_CHECKPOINT_INTERVAL_MINUTES <= 0:
    raise ValueError("tick rate and checkpoint interval must be positive")
if LIVE_STATE_INTERVAL_TICKS < 1 or TRAINING_MAX_THREADS < 1:
    raise ValueError("live-state interval and training thread cap must be positive")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(_json_value(payload), separators=(",", ":")) + "\n")
    os.replace(temp_path, path)


def latest_full_checkpoint() -> Path | None:
    paths = sorted(DATA.glob("checkpoint_full_*.json"))
    return paths[-1] if paths else None


def _rng_snapshot(envs: dict) -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch_cpu": torch.get_rng_state().tolist(),
        "gym": {role: env.unwrapped.np_random.bit_generator.state for role, env in envs.items()},
    }


def restore_rng(state: dict, envs: dict) -> None:
    if not state:
        return
    if state.get("python"):
        def as_tuple(value):
            return tuple(as_tuple(item) for item in value) if isinstance(value, list) else value
        random.setstate(as_tuple(state["python"]))
    if state.get("numpy"):
        name, keys, *rest = state["numpy"]
        np.random.set_state((name, np.asarray(keys, dtype=np.uint32), *rest))
    if state.get("torch_cpu"):
        torch.set_rng_state(torch.tensor(state["torch_cpu"], dtype=torch.uint8))
    for role, rng_state in state.get("gym", {}).items():
        if role in envs:
            envs[role].unwrapped.np_random.bit_generator.state = rng_state


def load_full_policies(path: Path, envs: dict) -> tuple[dict, dict]:
    payload = json.loads(path.read_text())
    with tempfile.TemporaryDirectory(prefix="flywiresim-resume-") as temp_dir:
        models = {}
        for role, env in envs.items():
            archive = base64.b64decode(payload["policies"][role])
            archive_path = Path(temp_dir) / f"{role}.zip"
            archive_path.write_bytes(archive)
            models[role] = PPO.load(archive_path, env=env, device="cpu")
    restore_rng(payload.get("rng", {}), envs)
    return models, payload


class TrainingRuntime:
    def __init__(self, models: dict, envs: dict, metrics: dict, seed: int,
                 epochs: list, output_path: Path, started_steps: int = 0,
                 checkpoint_interval_minutes: float = FULL_CHECKPOINT_INTERVAL_MINUTES):
        self.models = models
        self.envs = envs
        self.metrics = metrics
        self.seed = seed
        self.epochs = epochs
        self.output_path = output_path
        self.started_steps = started_steps
        self.steps_per_role = started_steps
        self.interval = checkpoint_interval_minutes * 60
        self.last_checkpoint = time.monotonic()
        self.last_live_tick = {"bat": -1, "moth": -1}
        self.recent_rewards = {role: {agent: deque(maxlen=10) for agent in ("bat", "moth")}
                               for role in envs}
        DATA.mkdir(parents=True, exist_ok=True)

    def write_live_state(self, role: str, info: dict) -> None:
        env = self.envs[role].unwrapped
        current_step = int(self.models[role].num_timesteps)
        if current_step - self.last_live_tick[role] < LIVE_STATE_INTERVAL_TICKS:
            return
        self.last_live_tick[role] = current_step
        rewards = self.recent_rewards[role]
        for agent in ("bat", "moth"):
            if f"{agent}_reward" in info:
                rewards[agent].append(float(info[f"{agent}_reward"]))
        moth_position = info.get("moth_position", env.moth.position.tolist())
        flame_position = info.get("flame_position", env.flame.tolist())
        flame_distance = float(np.linalg.norm(np.asarray(moth_position) - np.asarray(flame_position)))
        live = {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "active_training_role": role,
            "step": current_step,
            "episode": len(self.metrics[role].episodes) + 1,
            "tick": int(info.get("hunt_length", env.tick)),
            "bat": {"position": info.get("bat_position", env.bat.position.tolist()),
                    "velocity": info.get("bat_velocity", env.bat.velocity.tolist()),
                    "sonar_radius": float(env.cfg["sonar_range"]),
                    "jaw_position": info.get("mouth_position", env.mouth_position().tolist()),
                    "recent_reward": rewards["bat"][-1] if rewards["bat"] else 0.0},
            "moth": {"position": info.get("moth_position", env.moth.position.tolist()),
                     "velocity": info.get("moth_velocity", env.moth.velocity.tolist()),
                     "flame_distance": flame_distance,
                     "flame_position": flame_position,
                     "recent_reward": rewards["moth"][-1] if rewards["moth"] else 0.0},
            "caught": bool(info.get("caught", False)),
            "jaw_distance": float(info.get("jaw_distance", 0.0)),
            "jaw_contact_fraction": info.get("jaw_contact_fraction"),
            "jaw_contact_position": info.get("jaw_contact_position"),
            "sonar_occluded": bool(info.get("sonar_occluded", False)),
            "vision_occluded": bool(info.get("vision_occluded", False)),
            "occlusion_rate": info.get("occlusion_rate", {"bat": 0.0, "moth": 0.0}),
            "environment": {"obstacles": [
                {"x": float(item["xy"][0]), "y": float(item["xy"][1]),
                 "angle": float(np.arctan2(item["branch"][1], item["branch"][0]))}
                for item in env.obstacles],
                "terrain_displacement_scale": TERRAIN_DISPLACEMENT_SCALE,
                "terrain_noise_frequency": TERRAIN_NOISE_FREQUENCY},
            "last_10_rewards": {agent: list(queue) for agent, queue in rewards.items()},
            "attention_weights": [0.0] * int(env.observation_space.shape[0]),
            "workspace_weights": {"sensory": 0.0, "proprioception": 0.0,
                                  "self_model": 0.0, "belief_state": 0.0},
            "workspace_dominant": {"bat": None, "moth": None},
            "belief_state_estimated_position": [0.0, 0.0, 0.0],
            "recent_reward": rewards[role][-1] if rewards[role] else 0.0,
        }
        atomic_json(DATA / "live_state.json", live)

    def save_full_checkpoint(self, reason: str = "interval") -> Path:
        DATA.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="flywiresim-checkpoint-") as temp_dir:
            policies = {}
            for role, model in self.models.items():
                archive_path = Path(temp_dir) / f"{role}.zip"
                model.save(archive_path)
                policies[role] = base64.b64encode(archive_path.read_bytes()).decode("ascii")
        payload = {
            "format": 1,
            "training_mode": getattr(self, "training_mode", None),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "seed": self.seed,
            "steps_per_role": self.steps_per_role,
            "selfplay_steps": getattr(self, "selfplay_steps", 0),
            "policy_training_steps": {role: int(model.num_timesteps) for role, model in self.models.items()},
            "completed_episodes": {role: len(callback.episodes) for role, callback in self.metrics.items()},
            "current_environments": {role: {"tick": int(env.unwrapped.tick),
                                             "bat_position": env.unwrapped.bat.position.tolist(),
                                             "bat_velocity": env.unwrapped.bat.velocity.tolist(),
                                             "moth_position": env.unwrapped.moth.position.tolist(),
                                             "moth_velocity": env.unwrapped.moth.velocity.tolist()}
                                     for role, env in self.envs.items()},
            "policies": policies,
            "optimizer_state": "included in each Stable-Baselines3 policy archive",
            "behavioral_repertoires": {role: env.unwrapped.novelty_state()
                                        for role, env in self.envs.items()},
            "roles": {role: {"summary": {"episodes": len(callback.episodes)},
                             "episodes": callback.episodes}
                      for role, callback in self.metrics.items()},
            "epochs": self.epochs,
            "rng": _rng_snapshot(self.envs),
        }
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = DATA / f"checkpoint_full_{stamp}.json"
        atomic_json(destination, payload)
        checkpoints = sorted(DATA.glob("checkpoint_full_*.json"))
        for old in checkpoints[:-3]:
            old.unlink()
        self.last_checkpoint = time.monotonic()
        return destination

    def maybe_checkpoint(self, force: bool = False, reason: str = "interval") -> Path | None:
        if force or time.monotonic() - self.last_checkpoint >= self.interval:
            return self.save_full_checkpoint("shutdown" if force else reason)
        return None


class TickPacer:
    def __init__(self, ticks_per_second: float = TICKS_PER_SECOND,
                 battery_min_percent: int = TRAINING_MIN_BATTERY_PCT, enabled: bool = True):
        self.interval = 1.0 / ticks_per_second
        self.enabled = enabled
        self.next_tick = time.monotonic()
        self.battery_min_percent = max(0, min(100, battery_min_percent))
        self.last_battery_check = 0.0
        self.on_battery_low = False
        self.last_lag_warning = 0.0

    def _mac_battery_ok(self) -> bool:
        if time.monotonic() - self.last_battery_check < 60:
            return not self.on_battery_low
        self.last_battery_check = time.monotonic()
        try:
            status = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                                    text=True, timeout=3, check=False).stdout
        except (OSError, subprocess.TimeoutExpired):
            return True
        if "AC Power" in status or "No battery" in status:
            self.on_battery_low = False
            return True
        match = re.search(r"(\d+)%", status)
        self.on_battery_low = bool(match and int(match.group(1)) < self.battery_min_percent)
        if self.on_battery_low:
            print(f"Battery below {self.battery_min_percent}%; training paused until charged or plugged in.",
                  flush=True)
        return not self.on_battery_low

    def pace(self, stopping) -> bool:
        if not self.enabled:
            return not stopping()
        while not stopping():
            if self._mac_battery_ok():
                break
            time.sleep(1)
        if stopping():
            return False
        now = time.monotonic()
        delay = self.next_tick - now
        if delay > 0:
            time.sleep(delay)
        elif now - self.next_tick > self.interval and now - self.last_lag_warning > 10:
            print("WARNING: training is behind the real-time tick rate; continuing without dropping steps.",
                  flush=True)
            self.last_lag_warning = now
        self.next_tick = max(self.next_tick + self.interval, time.monotonic())
        return not stopping()
