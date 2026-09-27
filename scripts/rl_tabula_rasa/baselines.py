"""Small non-learning controls shared by benchmark runners."""

from __future__ import annotations

import math

import numpy as np


def random_action(rng: np.random.Generator) -> np.ndarray:
    """Uniform random continuous action in the environment's [-1, 1] space."""
    return rng.uniform(-1.0, 1.0, size=3).astype(np.float32)


def deterministic_pursuit_baseline(
    bat_position,
    moth_position,
    bat_velocity,
    yaw: float,
    pitch: float,
    *,
    max_yaw_delta: float = 0.12,
    max_pitch_delta: float = 0.08,
    bat_accel: float = 0.22,
    dt: float = 1.0,
    drag: float = 0.91,
    target_speed: float = 0.42,
) -> np.ndarray:
    """Turn directly toward the moth and hold a fixed target speed; no learning."""
    delta = np.asarray(moth_position, dtype=np.float32) - np.asarray(bat_position, dtype=np.float32)
    horizontal = math.hypot(float(delta[0]), float(delta[1]))
    target_yaw = math.atan2(float(delta[1]), float(delta[0]))
    target_pitch = math.atan2(float(delta[2]), max(horizontal, 1e-8))
    yaw_error = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
    yaw_action = np.clip(yaw_error / max_yaw_delta, -1.0, 1.0)
    pitch_action = np.clip((target_pitch - pitch) / max_pitch_delta, -1.0, 1.0)

    speed = float(np.linalg.norm(bat_velocity))
    needed_accel = max(0.0, target_speed - drag * speed) / dt
    thrust = np.clip(2.0 * needed_accel / max(bat_accel, 1e-8) - 1.0, -1.0, 1.0)
    return np.asarray((yaw_action, pitch_action, thrust), dtype=np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    assert np.all(np.abs(random_action(rng)) <= 1.0)
    action = deterministic_pursuit_baseline((0, 0, 1), (0, 1, 1), (0, 0, 0), 0, 0)
    assert action[0] > 0 and -1.0 <= action[2] <= 1.0
    print("baseline controls passed")
