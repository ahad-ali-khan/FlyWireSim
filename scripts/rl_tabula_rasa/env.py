"""Flat, boundary-only 3D arena. No scripted pursuit or evasion primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .environment_config import (JAW_CONTACT_RADIUS, OBSTACLE_DENSITY,
                                 OBSTACLE_JITTER, OBSTACLE_SEED,
                                 sample_obstacle_sites,
                                 TERRAIN_DISPLACEMENT_SCALE,
                                 TERRAIN_NOISE_FREQUENCY)

# SI units: convert the entire legacy per-tick flight envelope, not just the bat.
REFERENCE_DT = 0.16
PHYSICS_VERSION = "si_flight_synchronous_contact_v1"

CONFIG = {
    "arena_x": 3.5,
    "arena_y": 2.7,
    "floor_z": 0.3,
    "ceiling_z": 3.0,
    "boundary_falloff": 0.8,
    "boundary_k": 0.012 / REFERENCE_DT ** 2,
    "max_repulsion": 0.35 / REFERENCE_DT ** 2,
    "max_ticks": 70,
    "dt": REFERENCE_DT,
    "drag": 0.91,
    "bat_accel": 8.59375,
    "moth_accel": 0.15 / REFERENCE_DT ** 2,
    "bat_max_speed": 2.625,
    "moth_max_speed": 0.28 / REFERENCE_DT,
    "max_yaw_delta": 0.12,
    "max_pitch_delta": 0.08,
    "min_pitch": -1.2,
    "max_pitch": 1.2,
    "sonar_range": 6.0,
    "vision_range": 7.0,
    "jaw_forward_offset": 0.3337,
    "jaw_vertical_offset": 0.12,
    "jaw_contact_radius": JAW_CONTACT_RADIUS,
    "spawn_min_distance": 1.6,
    "bat_catch": 1.0,
    "bat_tick": -0.002,
    "bat_timeout": -0.5,
    "bat_distance_reduction": 0.1,
    "bat_heading_alignment_reward": 0.3,
    "reward_version": "terminal_safe_signed_potential_v2",
    "reward_gamma": 0.99,
    "bat_aligned_thrust_threshold": 0.3,
    "bat_aligned_thrust_reward": 0.08,
    "moth_survival": 1.0,
    "moth_tick": -0.01,
    "moth_caught": -0.5,
    "moth_bat_distance_gain": 0.015,
    "moth_flame_distance_reduction": 0.015,
    "natural_environment": False,
    "obstacle_density": OBSTACLE_DENSITY,
    "obstacle_jitter": OBSTACLE_JITTER,
    "obstacle_seed": OBSTACLE_SEED,
}


def unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-8 else np.zeros(3, dtype=np.float32)


def terrain_height(x: float, y: float) -> float:
    """Deterministic low-relief terrain matching the Blender displacement."""
    f = TERRAIN_NOISE_FREQUENCY
    return TERRAIN_DISPLACEMENT_SCALE * (
        0.5 * math.sin(x * f) * math.cos(y * f * 0.83)
        + 0.5 * math.sin((x + y) * f * 0.57)
    )


def _segment_distance(a0, a1, b0, b1) -> float:
    """Shortest distance between finite 3D segments."""
    u, v, w = a1 - a0, b1 - b0, a0 - b0
    aa, bb, cc = float(u @ u), float(u @ v), float(v @ v)
    dd, ee = float(u @ w), float(v @ w)
    denom = aa * cc - bb * bb
    s = 0.0 if denom < 1e-10 else float(np.clip((bb * ee - cc * dd) / denom, 0, 1))
    t = float(np.clip((bb * s + ee) / cc, 0, 1)) if cc > 1e-10 else 0.0
    s = float(np.clip((bb * t - dd) / aa, 0, 1)) if aa > 1e-10 else 0.0
    return float(np.linalg.norm((a0 + s * u) - (b0 + t * v)))


def _swept_contact(jaw_prev, jaw_curr, moth_prev, moth_curr, radius):
    """Synchronous linear sweep: earliest contact time, minimum gap, midpoint.

    Both bodies must occupy their positions at the SAME fraction of a tick.
    Independent segment closest points falsely catch paths crossing at different
    times. Relative motion reduces this to a segment versus a sphere.
    """
    jaw_prev = np.asarray(jaw_prev, dtype=np.float64)
    jaw_curr = np.asarray(jaw_curr, dtype=np.float64)
    moth_prev = np.asarray(moth_prev, dtype=np.float64)
    moth_curr = np.asarray(moth_curr, dtype=np.float64)
    jaw_delta, moth_delta = jaw_curr - jaw_prev, moth_curr - moth_prev
    offset = jaw_prev - moth_prev
    relative = jaw_delta - moth_delta
    aa = float(relative @ relative)
    bb = float(offset @ relative)
    cc = float(offset @ offset) - float(radius) ** 2
    closest_t = float(np.clip(-bb / aa, 0, 1)) if aa > 1e-12 else 0.0
    minimum_distance = float(np.linalg.norm(offset + closest_t * relative))
    if aa <= 1e-12 and cc > 0:
        return None, minimum_distance, None
    if minimum_distance > float(radius) + 1e-10:
        return None, minimum_distance, None
    contact_t = (0.0 if cc <= 0 else
                 float(np.clip((-bb - math.sqrt(max(0.0, bb * bb - aa * cc))) / aa, 0, 1)))
    jaw_point = jaw_prev + contact_t * jaw_delta
    moth_point = moth_prev + contact_t * moth_delta
    return contact_t, minimum_distance, ((jaw_point + moth_point) * 0.5).tolist()


def boundary_repulsion(position: np.ndarray, cfg: dict = CONFIG) -> np.ndarray:
    """Inverse-square inward force near the six faces of the box arena."""
    limits = ((-cfg["arena_x"], cfg["arena_x"]),
              (-cfg["arena_y"], cfg["arena_y"]),
              (cfg["floor_z"], cfg["ceiling_z"]))
    force = np.zeros(3, dtype=np.float32)
    for axis, (low, high) in enumerate(limits):
        for distance, sign in ((float(position[axis] - low), 1.0),
                               (float(high - position[axis]), -1.0)):
            if distance < cfg["boundary_falloff"]:
                force[axis] += sign * min(cfg["max_repulsion"],
                                          cfg["boundary_k"] / max(0.05, distance) ** 2)
    return force


@dataclass
class Body:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    yaw: float
    pitch: float


class HuntEnv(gym.Env):
    """One learner controls `role`; the other uses a network or random actions."""

    metadata = {"render_modes": []}

    def __init__(self, role: str = "bat", opponent=None, config: dict | None = None):
        super().__init__()
        if role not in ("bat", "moth"):
            raise ValueError("role must be bat or moth")
        self.role = role
        self.opponent = opponent
        self.cfg = CONFIG | (config or {})
        self.uses_occlusion = bool(self.cfg["natural_environment"])
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        # Heading closes the control state because yaw and pitch actions are deltas.
        # The relative velocity is needed to estimate an intercept rather than
        # chase the target's last measured position. It is masked out of range.
        # Phase 5 appends one explicit sensor-occlusion flag to each observation.
        self.observation_space = spaces.Box(-np.inf, np.inf,
            shape=((19 if role == "bat" else 23) if self.uses_occlusion
                   else (18 if role == "bat" else 22),), dtype=np.float32)
        self.bat = None
        self.moth = None
        self.flame = None
        self.tick = 0
        self.last_actions = None
        self.obstacles = []
        self.occlusion_ticks = {"bat": 0, "moth": 0}

    def _make_obstacles(self, run_seed):
        if not self.cfg["natural_environment"]:
            return []
        seed = self.cfg.get("obstacle_seed")
        sites = sample_obstacle_sites(run_seed if seed is None else int(seed),
                                      self.cfg["arena_x"], self.cfg["arena_y"],
                                      float(self.cfg["obstacle_density"]),
                                      float(self.cfg["obstacle_jitter"]))
        obstacles = []
        for x, y, angle in sites:
            x, y = float(x), float(y)
            branch = np.array([0.4 * math.cos(angle), 0.4 * math.sin(angle), 1.4],
                              dtype=np.float32)
            obstacles.append({"xy": np.array([x, y], dtype=np.float32), "radius": 0.12,
                              "height": 2.0, "branch": branch})
        return obstacles

    def _sensor_occluded(self, start: np.ndarray, end: np.ndarray) -> bool:
        if not self.obstacles:
            return False
        for obstacle in self.obstacles:
            x, y = obstacle["xy"]
            ground = self.cfg["floor_z"] + terrain_height(float(x), float(y))
            trunk_start = np.array([x, y, ground], dtype=np.float32)
            trunk_end = trunk_start + np.array([0, 0, obstacle["height"]], dtype=np.float32)
            if _segment_distance(start, end, trunk_start, trunk_end) <= obstacle["radius"]:
                return True
            branch_start = trunk_start + np.array([0, 0, 1.4], dtype=np.float32)
            branch_end = branch_start + obstacle["branch"]
            if _segment_distance(start, end, branch_start, branch_end) <= 0.08:
                return True
        return False

    def _spawn(self) -> np.ndarray:
        c = self.cfg
        return np.array([self.np_random.uniform(-c["arena_x"] + 0.8, c["arena_x"] - 0.8),
                         self.np_random.uniform(-c["arena_y"] + 0.8, c["arena_y"] - 0.8),
                         self.np_random.uniform(c["floor_z"] + 0.5, c["ceiling_z"] - 0.5)],
                        dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.tick = 0
        self.last_actions = None
        self.occlusion_ticks = {"bat": 0, "moth": 0}
        self.obstacles = self._make_obstacles(seed if seed is not None else
                                               int(self.np_random.integers(0, 2**31)))
        bat_pos = self._spawn()
        moth_pos = self._spawn()
        while np.linalg.norm(bat_pos - moth_pos) < self.cfg["spawn_min_distance"]:
            moth_pos = self._spawn()
        self.flame = self._spawn()
        self.bat = Body(bat_pos, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
                        float(self.np_random.uniform(-math.pi, math.pi)),
                        float(self.np_random.uniform(-0.35, 0.35)))
        self.moth = Body(moth_pos, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
                         float(self.np_random.uniform(-math.pi, math.pi)),
                         float(self.np_random.uniform(-0.35, 0.35)))
        return self.observe(self.role), {}

    def observe(self, role: str) -> np.ndarray:
        c = self.cfg
        remaining = (c["max_ticks"] - self.tick) / c["max_ticks"]
        heading = np.array([math.cos(self.bat.pitch) * math.cos(self.bat.yaw),
                            math.cos(self.bat.pitch) * math.sin(self.bat.yaw),
                            math.sin(self.bat.pitch)], dtype=np.float32)
        moth_heading = np.array([math.cos(self.moth.pitch) * math.cos(self.moth.yaw),
                                 math.cos(self.moth.pitch) * math.sin(self.moth.yaw),
                                 math.sin(self.moth.pitch)], dtype=np.float32)
        if role == "bat":
            delta = self.moth.position - self.mouth_position()
            distance = float(np.linalg.norm(delta))
            in_range = distance <= c["sonar_range"]
            occluded = in_range and self._sensor_occluded(self.mouth_position(), self.moth.position)
            hit = in_range and not occluded
            tail = [remaining, float(occluded)] if self.uses_occlusion else [remaining]
            return np.concatenate((self.bat.position, self.bat.velocity, heading,
                                   [float(hit), distance if hit else 0.0],
                                   unit(delta) if hit else np.zeros(3),
                                   self.moth.velocity - self.bat.velocity if hit else np.zeros(3),
                                   tail)).astype(np.float32)
        delta = self.bat.position - self.moth.position
        distance = float(np.linalg.norm(delta))
        in_range = distance <= c["vision_range"]
        occluded = in_range and self._sensor_occluded(self.moth.position, self.bat.position)
        visible = in_range and not occluded
        flame_delta = self.flame - self.moth.position
        tail = [float(np.linalg.norm(flame_delta)), remaining, float(occluded)] if self.uses_occlusion else [float(np.linalg.norm(flame_delta)), remaining]
        return np.concatenate((self.moth.position, self.moth.velocity, moth_heading,
                               [float(visible)], unit(delta) if visible else np.zeros(3),
                               [distance if visible else 0.0],
                               self.bat.velocity - self.moth.velocity if visible else np.zeros(3),
                               unit(flame_delta),
                               tail)).astype(np.float32)

    def mouth_position(self) -> np.ndarray:
        """Jaw point from the measured Blender lower-jaw bone offset.

        The rig's local +Y axis points forward. Its lower-jaw tail is offset by
        (0, 0.3337, -0.12) world units at the viewer's 0.18 bat scale.
        """
        c = self.cfg
        speed = float(np.linalg.norm(self.bat.velocity))
        if speed > 1e-6:
            heading = unit(self.bat.velocity)
            yaw = math.atan2(heading[1], heading[0])
            pitch = math.atan2(heading[2], max(1e-6, math.hypot(heading[0], heading[1])))
        else:
            yaw, pitch = self.bat.yaw, self.bat.pitch
        horizontal = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
        mouth_y = c["jaw_forward_offset"] * math.cos(pitch) + c["jaw_vertical_offset"] * math.sin(pitch)
        mouth_z = c["jaw_forward_offset"] * math.sin(pitch) - c["jaw_vertical_offset"] * math.cos(pitch)
        return self.bat.position + horizontal * mouth_y + np.array([0.0, 0.0, mouth_z], dtype=np.float32)

    def _move(self, body: Body, action: np.ndarray, species: str):
        c = self.cfg
        if species == "moth" and c.get("stationary_moth", False):
            body.velocity.fill(0)
            body.acceleration.fill(0)
            return
        yaw, pitch, thrust = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        body.yaw += float(yaw) * c["max_yaw_delta"]
        body.pitch = float(np.clip(body.pitch + pitch * c["max_pitch_delta"],
                                   c["min_pitch"], c["max_pitch"]))
        forward = np.array([math.cos(body.pitch) * math.cos(body.yaw),
                            math.cos(body.pitch) * math.sin(body.yaw),
                            math.sin(body.pitch)], dtype=np.float32)
        accel = c[f"{species}_accel"] * (float(thrust) + 1) / 2 * forward
        body.acceleration = accel + boundary_repulsion(body.position, c)
        body.velocity = c["drag"] * body.velocity + body.acceleration * c["dt"]
        speed = float(np.linalg.norm(body.velocity))
        if speed > c[f"{species}_max_speed"]:
            body.velocity *= c[f"{species}_max_speed"] / speed
        body.position += body.velocity * c["dt"]
        body.position[:] = np.clip(body.position,
                                   [-c["arena_x"], -c["arena_y"], c["floor_z"]],
                                   [c["arena_x"], c["arena_y"], c["ceiling_z"]])
        if self.cfg["natural_environment"]:
            floor = c["floor_z"] + terrain_height(float(body.position[0]), float(body.position[1]))
            body.position[2] = max(body.position[2], floor + 0.12)
            for obstacle in self.obstacles:
                delta = body.position[:2] - obstacle["xy"]
                distance = float(np.linalg.norm(delta))
                min_distance = obstacle["radius"] + 0.12
                if distance < min_distance and body.position[2] < c["floor_z"] + obstacle["height"]:
                    normal = unit(np.array([delta[0], delta[1], 0], dtype=np.float32))
                    if not np.any(normal):
                        normal = np.array([1, 0, 0], dtype=np.float32)
                    body.position[:2] = obstacle["xy"] + normal[:2] * min_distance
                    inward = float(body.velocity[:2] @ normal[:2])
                    if inward < 0:
                        body.velocity[:2] -= inward * normal[:2]

    def step(self, action):
        other = "moth" if self.role == "bat" else "bat"
        other_obs = self.observe(other)
        other_action = (self.action_space.sample() if self.opponent is None
                        else self.opponent.predict(other_obs, deterministic=False)[0])
        return HuntEnv.step_with_actions(self, {self.role: action, other: other_action})

    def step_with_actions(self, actions):
        """Advance one shared hunt with explicit actions for both agents."""
        self.last_actions = actions
        old_sensor = self.observe("bat")
        jaw_prev = self.mouth_position().copy()
        moth_prev = self.moth.position.copy()
        old_body_distance = float(np.linalg.norm(self.bat.position - self.moth.position))
        old_flame = float(np.linalg.norm(self.moth.position - self.flame))
        self._move(self.bat, actions["bat"], "bat")
        self._move(self.moth, actions["moth"], "moth")
        self.tick += 1
        jaw_curr = self.mouth_position().copy()
        moth_curr = self.moth.position.copy()
        bat_sensor = self.observe("bat")
        moth_sensor = self.observe("moth")
        sonar_occluded = bool(self.uses_occlusion and bat_sensor[-1] > 0.5)
        vision_occluded = bool(self.uses_occlusion and moth_sensor[-1] > 0.5)
        self.occlusion_ticks["bat"] += int(sonar_occluded)
        self.occlusion_ticks["moth"] += int(vision_occluded)
        new_distance = float(np.linalg.norm(self.mouth_position() - self.moth.position))
        new_body_distance = float(np.linalg.norm(self.bat.position - self.moth.position))
        new_flame = float(np.linalg.norm(self.moth.position - self.flame))
        contact_fraction, swept_distance, contact_position = _swept_contact(
            jaw_prev, jaw_curr, moth_prev, moth_curr, self.cfg["jaw_contact_radius"])
        caught = contact_fraction is not None
        timeout = self.tick >= self.cfg["max_ticks"] and not caught
        c = self.cfg
        bat_reward = c["bat_tick"]
        bat_alignment = None
        bat_alignment_reward = 0.0
        bat_aligned_thrust_reward = 0.0
        if bat_sensor[9] > 0.5:
            heading = np.array([math.cos(self.bat.pitch) * math.cos(self.bat.yaw),
                                math.cos(self.bat.pitch) * math.sin(self.bat.yaw),
                                math.sin(self.bat.pitch)], dtype=np.float32)
            bat_alignment = float(heading @ bat_sensor[11:14])
        # Reward improvement, not dwelling in an aligned pose. With terminal
        # potential zero, discounted shaping telescopes to -Phi(initial), so it
        # cannot turn a failed hunt into a better objective than a catch.
        def potentials(sensor):
            if sensor[9] <= 0.5:
                return 0.0, 0.0
            alignment = float(sensor[6:9] @ sensor[11:14])
            # Signed alignment also gives steering feedback when facing away.
            # Retain the old definition for explicitly versioned evaluations.
            if c["reward_version"] == "terminal_safe_potential_v1":
                alignment = max(0.0, alignment)
            return (-c["bat_distance_reduction"] * float(sensor[10]),
                    c["bat_heading_alignment_reward"] * alignment)

        old_distance_potential, old_alignment_potential = potentials(old_sensor)
        next_distance_potential, next_alignment_potential = ((0.0, 0.0) if caught or timeout
                                                             else potentials(bat_sensor))
        bat_distance_reward = c["reward_gamma"] * next_distance_potential - old_distance_potential
        bat_alignment_reward = c["reward_gamma"] * next_alignment_potential - old_alignment_potential
        bat_reward += bat_distance_reward + bat_alignment_reward
        moth_reward = (c["moth_tick"] + c["moth_bat_distance_gain"] * (new_body_distance - old_body_distance)
                       + c["moth_flame_distance_reduction"] * (old_flame - new_flame))
        if caught:
            bat_reward += c["bat_catch"]
            moth_reward += c["moth_caught"]
        elif timeout:
            bat_reward += c["bat_timeout"]
            moth_reward += c["moth_survival"]
        return (self.observe(self.role), bat_reward if self.role == "bat" else moth_reward,
                caught or timeout, False,
                {"caught": caught, "timeout": timeout, "hunt_length": self.tick,
                 "bat_reward": bat_reward, "moth_reward": moth_reward,
                 "bat_alignment": bat_alignment,
                 "bat_alignment_reward": bat_alignment_reward,
                 "bat_distance_reward": bat_distance_reward,
                 "bat_aligned_thrust_reward": bat_aligned_thrust_reward,
                 "jaw_distance": new_distance, "body_distance": new_body_distance,
                 "swept_jaw_distance": swept_distance,
                 "jaw_contact_fraction": contact_fraction,
                 "jaw_contact_position": contact_position,
                 "bat_position": self.bat.position.tolist(),
                 "moth_position": self.moth.position.tolist(),
                 "bat_velocity": self.bat.velocity.tolist(),
                 "moth_velocity": self.moth.velocity.tolist(),
                 "flame_position": self.flame.tolist(),
                "mouth_position": self.mouth_position().tolist(),
                "sonar_occluded": sonar_occluded,
                "vision_occluded": vision_occluded,
                "occlusion_rate": {role: ticks / self.tick for role, ticks in self.occlusion_ticks.items()},
                "obstacle_count": len(self.obstacles)})


def assert_3d_motion():
    flight = HuntEnv("bat")
    flight.reset(seed=7)
    flight.bat.position[:] = (0, 0, 1.0)
    flight.bat.velocity[:] = 0
    start_z = float(flight.bat.position[2])
    for _ in range(15):
        flight._move(flight.bat, np.array([0, 1, 1]), "bat")
    assert flight.bat.position[2] - start_z > 0.1, "3D flight regressed to a flat plane"
