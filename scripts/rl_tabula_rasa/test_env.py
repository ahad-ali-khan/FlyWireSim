"""Minimal regression checks for Phase 0 physics, observations, and gate."""

import numpy as np

from .env import (CONFIG, HuntEnv, assert_3d_motion, boundary_repulsion,
                  terrain_height, _swept_contact)
from .train import gate


def test_environment():
    # Their paths intersect within a tick although the endpoints are apart.
    fraction, gap, point = _swept_contact(
        (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), 0.1)
    assert fraction is not None and np.isclose(fraction, 0.5 - 0.1 / np.sqrt(8))
    assert gap < 1e-6 and np.isfinite(point).all()
    # Geometric intersection at different times must NOT count as a catch.
    fraction, gap, _ = _swept_contact(
        (-1, 0, 0), (1, 0, 0), (0, -0.5, 0), (0, 1.5, 0), 0.1)
    assert fraction is None and np.isclose(gap, np.sqrt(0.125))
    assert _swept_contact((0, 0, 0), (0, 0, 0), (1, 0, 0), (1, 0, 0), .1)[0] is None
    assert _swept_contact((0, 0, 0), (0, 0, 0), (.05, 0, 0), (.05, 0, 0), .1)[0] == 0
    assert _swept_contact((0, 0, 0), (0, 0, 0), (.10000000001, 0, 0), (.10000000001, 0, 0), .1)[0] is None
    swept = HuntEnv("bat")
    swept.reset(seed=17)
    swept.bat.position[:] = (-1.3337, 0.0, 1.12)
    swept.bat.velocity[:] = 0
    swept.bat.yaw = swept.bat.pitch = 0.0
    swept.moth.position[:] = (0.0, -1.0, 1.0)

    def scripted_tick(body, _action, species):
        if species == "bat":
            body.position += (2.0, 0.0, 0.0)
            body.velocity[:] = (1.0, 0.0, 0.0)
        else:
            body.position += (0.0, 2.0, 0.0)
            body.velocity[:] = (0.0, 1.0, 0.0)

    swept._move = scripted_tick
    _, _, done, _, contact_info = swept.step_with_actions({
        "bat": np.zeros(3, dtype=np.float32), "moth": np.zeros(3, dtype=np.float32)})
    assert done and contact_info["caught"] and not contact_info["timeout"]
    contact_t = 0.5 - CONFIG["jaw_contact_radius"] / np.sqrt(8)
    assert np.isclose(contact_info["jaw_contact_fraction"], contact_t)
    assert np.allclose(contact_info["jaw_contact_position"], (contact_t - .5, contact_t - .5, 1))

    def alignment_trial(yaw, sonar_range=6.0):
        env = HuntEnv("bat", config={"stationary_moth": True,
                                     "sonar_range": sonar_range})
        env.reset(seed=21)
        env.bat.position[:] = (0, 0, 1.5)
        env.bat.velocity[:] = 0
        env.bat.yaw = yaw
        env.bat.pitch = 0
        env.moth.position[:] = (2, 0, 1.5)
        return env.step_with_actions({
            "bat": np.array([0, 0, 1], dtype=np.float32),
            "moth": np.zeros(3, dtype=np.float32)})[4]

    aligned = alignment_trial(0)
    assert aligned["bat_alignment"] > 0.99
    assert aligned["bat_aligned_thrust_reward"] == 0
    assert aligned["bat_alignment_reward"] < .01  # no rent for staying aligned
    marginal_alignment = alignment_trial(0.9)
    assert 0.3 < marginal_alignment["bat_alignment"] < 0.5
    assert marginal_alignment["bat_aligned_thrust_reward"] == 0
    away = alignment_trial(np.pi)
    assert away["bat_alignment"] < 0
    assert abs(away["bat_alignment_reward"]) < .01
    assert away["bat_aligned_thrust_reward"] == 0
    unseen = alignment_trial(0, sonar_range=0.1)
    assert unseen["bat_alignment"] is None
    assert unseen["bat_alignment_reward"] == unseen["bat_aligned_thrust_reward"] == 0

    still = HuntEnv("moth", config={"stationary_moth": True})
    still.reset(seed=19)
    still_position = still.moth.position.copy()
    still._move(still.moth, np.ones(3, dtype=np.float32), "moth")
    assert np.array_equal(still.moth.position, still_position)
    assert np.array_equal(still.moth.velocity, np.zeros(3, dtype=np.float32))
    integrated = HuntEnv("bat")
    integrated.reset(seed=12)
    integrated.bat.position[:] = (0, 0, 1.5)
    integrated.bat.velocity[:] = 0
    integrated.bat.yaw = integrated.bat.pitch = 0
    integrated._move(integrated.bat, np.array([0, 0, 1], dtype=np.float32), "bat")
    assert np.isclose(integrated.bat.velocity[0], CONFIG["bat_accel"] * CONFIG["dt"])
    assert np.isclose(integrated.bat.position[0],
                      CONFIG["bat_accel"] * CONFIG["dt"] ** 2)
    for role, size in (("bat", 18), ("moth", 22)):
        env = HuntEnv(role)
        obs, _ = env.reset(seed=11)
        assert obs.shape == (size,)
        assert np.linalg.norm(env.bat.position - env.moth.position) >= CONFIG["spawn_min_distance"]
        for _ in range(CONFIG["max_ticks"]):
            obs, _, done, truncated, info = env.step(np.zeros(3, dtype=np.float32))
            assert obs.shape == (size,) and np.isfinite(obs).all() and not truncated
            if done:
                assert info["caught"] != info["timeout"]
                break
    assert boundary_repulsion(np.array([3.4, 0, 1.5]))[0] < 0
    assert boundary_repulsion(np.array([0, 0, 1.5]))[0] == 0
    assert not gate([])["passed"]
    assert_3d_motion()
    bat = HuntEnv("bat")
    bat.reset(seed=5)
    bat.bat.position[:] = 0
    bat.bat.velocity[:] = 0
    bat.bat.yaw = 0
    bat.bat.pitch = 0
    assert np.allclose(bat.mouth_position(), (CONFIG["jaw_forward_offset"], 0,
                                              -CONFIG["jaw_vertical_offset"]))
    bat.moth.velocity[:] = (0.1, 0.2, -0.1)
    assert np.allclose(bat.observe("bat")[14:17], bat.moth.velocity - bat.bat.velocity)
    outdoor = HuntEnv("bat", config={"natural_environment": True})
    outdoor.reset(seed=73)
    assert outdoor.observe("bat").shape == (19,)
    first_layout = [item["xy"].copy() for item in outdoor.obstacles]
    outdoor.reset(seed=73)
    assert len(outdoor.obstacles) == 3
    assert all(np.allclose(a, b["xy"]) for a, b in zip(first_layout, outdoor.obstacles))
    assert abs(terrain_height(0, 0)) < 1e-9
    outdoor.obstacles = [{"xy": np.array([0.0, 0.0], dtype=np.float32),
                          "radius": 0.12, "height": 2.0,
                          "branch": np.array([0.4, 0.0, 0.0], dtype=np.float32)}]
    assert outdoor._sensor_occluded(np.array([-1, 0, 1], dtype=np.float32),
                                    np.array([1, 0, 1], dtype=np.float32))


if __name__ == "__main__":
    test_environment()
    print("Phase 0 checks passed")
