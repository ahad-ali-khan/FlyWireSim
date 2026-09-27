from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT / "data/results/brian2cpp_t0.1s_n1.parquet"
BAT_OBJ = ROOT / "assets/models/creatures/bat/Bat.obj"
MOTH_OBJ = ROOT / "assets/models/creatures/moth/Moth.obj"


def brain_trace() -> np.ndarray:
    if not TRACE.exists():
        raise FileNotFoundError(
            f"{TRACE} is missing. Run first: uv run flywiresim --t-run 0.1"
        )
    spikes = pd.read_parquet(TRACE, columns=["time_ms", "flywire_id"])
    # These are named SEZ motor groups shipped with the upstream FlyWire repo.
    # Their differential activity is the first bat/moth steering signal.
    left_ids = {720575940614763666}
    right_ids = {720575940631997032, 720575940647030324}
    bins = np.linspace(0.0, 100.0, 101)
    left, _ = np.histogram(
        spikes.loc[spikes.flywire_id.isin(left_ids), "time_ms"], bins
    )
    right, _ = np.histogram(
        spikes.loc[spikes.flywire_id.isin(right_ids), "time_ms"], bins
    )
    signal = (right - left).astype(float)
    if np.max(np.abs(signal), initial=0.0) == 0:
        return np.zeros(100)
    return np.clip(signal / np.max(np.abs(signal)), -1.0, 1.0)


def xml() -> str:
    return f"""
<mujoco model="flywire-world">
  <compiler angle="degree" meshdir="{ROOT / 'assets/models/creatures'}" />
  <option timestep="0.01" gravity="0 0 -9.81" />
  <asset>
    <mesh name="bat_mesh" file="bat/VampireBat.obj" scale="0.18 0.18 0.18" />
    <mesh name="moth_mesh" file="moth/Moth.obj" scale="0.12 0.12 0.12" />
    <material name="bat_material" rgba="0.09 0.12 0.18 1" />
    <material name="moth_material" rgba="0.84 0.26 0.62 1" />
    <material name="ground_material" rgba="0.08 0.13 0.11 1" />
    <material name="food_material" rgba="0.95 0.62 0.08 1" />
  </asset>
  <worldbody>
    <light pos="0 0 8" dir="0 0 -1" diffuse="1 1 1" />
    <geom name="ground" type="plane" size="10 10 .1" material="ground_material" />
    <geom name="food" type="cylinder" pos="0 0 0.18" size="0.35 0.18" material="food_material" />
    <body name="bat" pos="-3 -1 2.8">
      <joint name="bat_x" type="slide" axis="1 0 0" />
      <joint name="bat_y" type="slide" axis="0 1 0" />
      <joint name="bat_z" type="slide" axis="0 0 1" />
      <joint name="bat_yaw" type="hinge" axis="0 0 1" />
      <geom name="bat_visual" type="mesh" mesh="bat_mesh" material="bat_material" contype="0" conaffinity="0" />
      <geom name="bat_body" type="ellipsoid" size="0.18 0.12 0.12" rgba="0.12 0.16 0.24 0.25" contype="1" conaffinity="1" />
    </body>
    <body name="moth" pos="3 1 1.7">
      <joint name="moth_x" type="slide" axis="1 0 0" />
      <joint name="moth_y" type="slide" axis="0 1 0" />
      <joint name="moth_z" type="slide" axis="0 0 1" />
      <joint name="moth_yaw" type="hinge" axis="0 0 1" />
      <geom name="moth_visual" type="mesh" mesh="moth_mesh" material="moth_material" contype="0" conaffinity="0" />
      <geom name="moth_body" type="ellipsoid" size="0.10 0.08 0.08" rgba="0.92 0.30 0.68 0.25" contype="1" conaffinity="1" />
    </body>
  </worldbody>
</mujoco>
"""


def run(seconds: float, render: bool) -> None:
    signal = brain_trace()
    model = mujoco.MjModel.from_xml_string(xml())
    data = mujoco.MjData(model)
    bat = model.body("bat").id
    moth = model.body("moth").id
    print(
        f"FlyWire world | {model.nbody - 1} agents | "
        f"{len(signal)} brain readout frames | {TRACE.name}"
    )

    viewer = None
    if render:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = [0, 0, 1.2]
        viewer.cam.distance = 9
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

    steps = max(1, int(seconds / model.opt.timestep))
    for step in range(steps):
        t = step * model.opt.timestep
        i = min(len(signal) - 1, int(t / max(seconds, 0.001) * len(signal)))
        brain = float(signal[i])
        phase = t * 2.0
        bat_pos = data.xpos[bat].copy()
        moth_pos = data.xpos[moth].copy()
        bat_target = np.array([0.0, 0.0, 1.3])
        moth_target = np.array([0.0, 0.0, 0.8])
        bat_pos[:2] += 0.018 * (bat_target[:2] - bat_pos[:2])
        moth_pos[:2] += 0.014 * (moth_target[:2] - moth_pos[:2])
        bat_pos[1] += 0.012 * brain
        moth_pos[1] -= 0.016 * brain
        bat_pos[2] = 2.8 + 0.15 * np.sin(phase)
        moth_pos[2] = 1.7 + 0.08 * np.sin(phase * 1.7)
        data.qpos[0:3] = bat_pos - np.array([-3.0, -1.0, 2.8])
        data.qpos[3] = np.arctan2(bat_target[1] - bat_pos[1], bat_target[0] - bat_pos[0])
        data.qpos[4:7] = moth_pos - np.array([3.0, 1.0, 1.7])
        data.qpos[7] = np.arctan2(moth_target[1] - moth_pos[1], moth_target[0] - moth_pos[0])
        mujoco.mj_forward(model, data)
        if viewer:
            viewer.sync()
            time.sleep(model.opt.timestep)

    if viewer:
        print("World running — close the MuJoCo window to exit")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
        viewer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the FlyWire 3D world")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    run(args.seconds, render=not args.headless)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
