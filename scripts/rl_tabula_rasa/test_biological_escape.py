"""Smoke checks for the biological escape circuit's pitch readout."""

import numpy as np

from .biological_escape import FlyWireEscapeCircuit, PITCH_ESCAPE_BIAS_RADIANS
from .env import HuntEnv


def test_pitch_bias_and_edge():
    assert PITCH_ESCAPE_BIAS_RADIANS == 0.30
    env = HuntEnv(config={"natural_environment": True, "obstacle_seed": None})
    env.reset(seed=23)
    env.moth.position[:] = (0.0, 0.0, 1.0)
    env.bat.position[:] = (0.8, 0.0, 1.0)
    env.bat.velocity[:] = (1.0, 0.0, 0.0)
    action, state = FlyWireEscapeCircuit().action(env)
    assert state["pitch_escape_bias_radians"] > 0.0
    assert action.shape == (3,) and np.all(np.abs(action) <= 1.0)


if __name__ == "__main__":
    test_pitch_bias_and_edge()
    print("biological escape checks passed")
