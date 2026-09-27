"""Connectome-grounded visual looming escape controller for the moth.

This is a small causal circuit, not a claim that the whole FlyWire brain is
being simulated in the arena.  The v783 edge table contains an LPLC2 -> DNp01
(Giant Fibre) connection with three synapses.  We preserve that named pathway
as a bilateral two-column leaky integrate-and-fire circuit: looming input
drives LPLC2, LPLC2 spikes drive DNp01, and DNp01 activity drives the moth's
continuous action vector.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


# FlyWire FAFB v783 root IDs used by the checked-in local connectivity table.
LPLC2_ROOT_ID = 720575940640302389
GIANT_FIBER_DNP01_ROOT_ID = 720575940622838154
LPLC2_TO_GIANT_FIBER_SYNAPSES = 3
MOTH_VISUAL_RADIUS = 0.12
LOOMING_ANGULAR_RATE_SCALE = 0.15
NEURAL_SUBSTEP_DT = 0.005
NEURAL_SUBSTEPS = 32  # 32 × 5 ms = one 160 ms environment tick.
LPLC2_MEMBRANE_TAU = 0.020
GIANT_FIBER_MEMBRANE_TAU = 0.005
PHOTORECEPTOR_TO_LOOMING_GAIN = 2.8
GIANT_FIBER_THRESHOLD = 0.72
SPIKE_RATE_DECAY = 0.82
PITCH_ESCAPE_BIAS_RADIANS = 0.30


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else np.zeros(3, dtype=np.float32)


class FlyWireEscapeCircuit:
    """Bilateral LPLC2→DNp01 escape circuit with an action readout."""

    def __init__(self, *, seed: int = 0, trace_path: str | Path | None = None):
        # Kept in the constructor for stable experiment manifests.  The
        # circuit itself is deterministic for a given arena trajectory.
        self.seed = seed
        self.reset()

    def reset(self):
        self.lplc2_voltage = np.zeros(2, dtype=np.float32)
        self.giant_fiber_voltage = np.zeros(2, dtype=np.float32)
        self.lplc2_spikes = np.zeros(2, dtype=np.float32)
        self.giant_fiber_spikes = np.zeros(2, dtype=np.float32)
        self.giant_fiber_rate = np.zeros(2, dtype=np.float32)
        self.last_state = {
            "looming": 0.0, "closing_speed": 0.0, "time_to_contact": None,
            "lplc2_spikes": [0, 0], "giant_fiber_spikes": [0, 0],
            "lplc2_spike_count": [0, 0], "giant_fiber_spike_count": [0, 0],
            "lplc2_rate": [0.0, 0.0], "giant_fiber_rate": [0.0, 0.0],
            "circuit": "LPLC2 -> DNp01/Giant Fibre -> motor readout",
            "lplc2_root_id": LPLC2_ROOT_ID,
            "giant_fiber_root_id": GIANT_FIBER_DNP01_ROOT_ID,
            "connectome_synapses": LPLC2_TO_GIANT_FIBER_SYNAPSES,
        }

    def _looming(self, env):
        delta = env.bat.position - env.moth.position
        distance = float(np.linalg.norm(delta))
        relative_velocity = env.bat.velocity - env.moth.velocity
        direction = _unit(delta)
        closing_speed = max(0.0, float(relative_velocity @ direction))
        # Angular-size expansion is the visual looming cue.  This is the
        # small-target approximation d(theta)/dt ≈ r*v_close/d², rather than
        # a distance-only trigger.  The radius is an arena-scale visual proxy
        # for the imported moth, not a claim about a neuron-level measurement.
        angular_rate = (MOTH_VISUAL_RADIUS * closing_speed /
                        max(distance * distance, 1e-6))
        looming = float(np.clip(angular_rate / LOOMING_ANGULAR_RATE_SCALE, 0, 1))
        ttc = distance / max(closing_speed, 1e-6) if closing_speed > 1e-6 else None
        return delta, distance, closing_speed, angular_rate, looming, ttc

    def _step_neurons(self, left_drive: float, right_drive: float):
        drives = np.asarray([left_drive, right_drive], dtype=np.float32)
        lplc2_decay = math.exp(-NEURAL_SUBSTEP_DT / LPLC2_MEMBRANE_TAU)
        giant_fiber_decay = math.exp(-NEURAL_SUBSTEP_DT / GIANT_FIBER_MEMBRANE_TAU)
        lplc2_train, giant_fiber_train = [], []
        for _ in range(NEURAL_SUBSTEPS):
            self.lplc2_voltage = (lplc2_decay * self.lplc2_voltage +
                                  PHOTORECEPTOR_TO_LOOMING_GAIN * (1.0 - lplc2_decay) * drives)
            lplc2_spikes = (self.lplc2_voltage >= 1.0).astype(np.float32)
            self.lplc2_voltage[lplc2_spikes > 0] = 0.0
            synaptic_input = (LPLC2_TO_GIANT_FIBER_SYNAPSES / 3.0) * lplc2_spikes
            self.giant_fiber_voltage = (giant_fiber_decay * self.giant_fiber_voltage +
                                        synaptic_input)
            giant_fiber_spikes = (self.giant_fiber_voltage >= GIANT_FIBER_THRESHOLD).astype(np.float32)
            self.giant_fiber_voltage[giant_fiber_spikes > 0] = 0.0
            lplc2_train.append(lplc2_spikes.astype(int).tolist())
            giant_fiber_train.append(giant_fiber_spikes.astype(int).tolist())
        self.lplc2_spikes = np.asarray(lplc2_train[-1], dtype=np.float32)
        self.giant_fiber_spikes = np.asarray(giant_fiber_train[-1], dtype=np.float32)
        self.lplc2_rate = np.mean(lplc2_train, axis=0).astype(np.float32)
        self.giant_fiber_rate = np.mean(giant_fiber_train, axis=0).astype(np.float32)
        return lplc2_train, giant_fiber_train

    def action(self, env) -> tuple[np.ndarray, dict]:
        delta, distance, closing_speed, angular_rate, looming, ttc = self._looming(env)
        direction = _unit(delta)
        # Moth-local right vector.  The two visual channels are spatially
        # separated; stronger right-channel drive means threat on the right.
        heading = np.asarray([
            math.cos(env.moth.pitch) * math.cos(env.moth.yaw),
            math.cos(env.moth.pitch) * math.sin(env.moth.yaw),
            math.sin(env.moth.pitch),
        ], dtype=np.float32)
        right = np.asarray([-math.sin(env.moth.yaw), math.cos(env.moth.yaw), 0.0], dtype=np.float32)
        lateral = float(np.clip(direction @ right, -1, 1))
        left_drive = looming * float(np.clip(.5 - .5 * lateral, 0, 1))
        right_drive = looming * float(np.clip(.5 + .5 * lateral, 0, 1))
        lplc2_train, giant_fiber_train = self._step_neurons(left_drive, right_drive)

        left_gf_rate = float(self.giant_fiber_rate[0])
        right_gf_rate = float(self.giant_fiber_rate[1])
        escape_strength = float(np.clip((left_gf_rate + right_gf_rate) / 2.0, 0, 1))

        # Explicit spike-to-action readout.  Bilateral asymmetry turns away
        # from the more active side; total GF activity drives thrust.  The
        # gains and the small pitch bias are engineering choices, not
        # biological parameters measured from the connectome.
        escape_yaw = np.clip(2.0 * (left_gf_rate - right_gf_rate), -1, 1)
        # The environment caps one tick at max_pitch_delta radians. Express
        # the requested 0.30 rad escape bias in action units so it accumulates
        # over several ticks without bypassing that physics limit.
        pitch_bias_radians = PITCH_ESCAPE_BIAS_RADIANS * escape_strength
        escape_pitch = np.clip(pitch_bias_radians / env.cfg["max_pitch_delta"], -1, 1)
        escape_thrust = np.clip(-0.15 + 1.9 * escape_strength, -1, 1)

        # When the GF is quiet, retain the existing flame attraction as the
        # moth's baseline drive.  The looming circuit takes over as GF rate
        # rises; this keeps the ecological light cue without hiding the
        # causal spike-to-motor mapping above.
        light = _unit(env.flame - env.moth.position)
        light_yaw = np.clip(float(light @ right) / max(.05, float(light @ heading)), -1, 1)
        light_pitch = np.clip(float(light[2]) - float(heading[2]), -1, 1)
        action = np.asarray([
            np.clip(escape_yaw + (1.0 - escape_strength) * 0.35 * light_yaw, -1, 1),
            np.clip(escape_pitch + (1.0 - escape_strength) * 0.18 * light_pitch, -1, 1),
            np.clip(escape_thrust + (1.0 - escape_strength) * (-0.15), -1, 1),
        ], dtype=np.float32)
        self.last_state = {
            "looming": looming, "angular_size_rate": angular_rate,
            "closing_speed": closing_speed,
            "time_to_contact": ttc, "distance": distance,
            "lplc2_drive": [left_drive, right_drive],
            "lplc2_spikes": self.lplc2_spikes.astype(int).tolist(),
            "giant_fiber_spikes": self.giant_fiber_spikes.astype(int).tolist(),
            "lplc2_spike_count": np.sum(lplc2_train, axis=0).astype(int).tolist(),
            "giant_fiber_spike_count": np.sum(giant_fiber_train, axis=0).astype(int).tolist(),
            "lplc2_spike_train": lplc2_train,
            "giant_fiber_spike_train": giant_fiber_train,
            "lplc2_rate": self.lplc2_rate.astype(float).tolist(),
            "giant_fiber_rate": self.giant_fiber_rate.astype(float).tolist(),
            "pitch_escape_bias_radians": pitch_bias_radians,
            "action_mapping": {
                "yaw": "2*(left_gf_rate-right_gf_rate) + quiet_gf_flame_bias",
                "pitch": "clip(0.30 rad*mean_gf_rate / max_pitch_delta) + quiet_gf_flame_bias",
                "thrust": "-0.15 + 1.9*mean_gf_rate + quiet_gf_flame_bias",
                "note": "engineering readout; not directly measured biological motor weights",
            },
            "action": action.astype(float).tolist(),
            "circuit": "LPLC2 -> DNp01/Giant Fibre -> motor readout",
            "lplc2_root_id": LPLC2_ROOT_ID,
            "giant_fiber_root_id": GIANT_FIBER_DNP01_ROOT_ID,
            "connectome_synapses": LPLC2_TO_GIANT_FIBER_SYNAPSES,
        }
        return action, dict(self.last_state)


def validate_connectome_edge(path: str | Path) -> dict:
    """Validate the claimed edge without loading the full table into this module."""
    import pandas as pd

    edges = pd.read_parquet(path, columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"])
    match = edges[(edges.Presynaptic_ID == LPLC2_ROOT_ID) &
                  (edges.Postsynaptic_ID == GIANT_FIBER_DNP01_ROOT_ID)]
    synapses = int(match.Connectivity.sum())
    if synapses != LPLC2_TO_GIANT_FIBER_SYNAPSES:
        raise ValueError(f"expected 3 LPLC2→DNp01 synapses, found {synapses}")
    return {"presynaptic": LPLC2_ROOT_ID, "postsynaptic": GIANT_FIBER_DNP01_ROOT_ID,
            "synapses": synapses}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(validate_connectome_edge(root / "data/2025_Connectivity_783.parquet")))
