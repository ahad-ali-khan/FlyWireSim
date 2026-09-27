"""Launch the textured Blender viewer for live training state or recorded episodes.

Usage: uv run python scripts/viewer.py --mode live|replay [--replay-file PATH] [--speed 1.0]
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("live", "replay"), default="replay")
    parser.add_argument("--replay-file", type=Path, default=ROOT / "results/phase2_replay.json")
    parser.add_argument("--live-stream-file", type=Path, default=ROOT / "data/live_state_stream.jsonl")
    parser.add_argument("--live-state-file", type=Path, default=ROOT / "data/live_state.json")
    parser.add_argument("--learning-curve", type=Path)
    parser.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    parser.add_argument("--hud", choices=("cinematic", "full-data"), default="full-data")
    parser.add_argument("--camera", choices=("wide", "chase", "hero"), default="chase")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("speed must be greater than zero")
    replay = args.replay_file.resolve()
    if args.mode == "replay" and not replay.is_file():
        parser.error(f"replay file missing: {replay}; run export_replay first")
    if not BLENDER.is_file():
        parser.error(f"Blender not found at {BLENDER}")
    env = os.environ.copy()
    env["FLYWIRE_VIEWER_MODE"] = args.mode
    env["FLYWIRE_REPLAY_SPEED"] = str(args.speed)
    env["FLYWIRE_HUD_MODE"] = args.hud
    env["FLYWIRE_CAMERA_PRESET"] = args.camera
    if args.learning_curve:
        env["FLYWIRE_LEARNING_CURVE_FILE"] = str(args.learning_curve.resolve())
    if args.mode == "replay":
        env.pop("FLYWIRE_LIVE_STATE_FILE", None)
        env["FLYWIRE_REPLAY_FILE"] = str(replay)
    else:
        env.pop("FLYWIRE_REPLAY_FILE", None)
        env["FLYWIRE_LIVE_STATE_FILE"] = str(args.live_state_file.resolve())
        env["FLYWIRE_LIVE_STREAM_FILE"] = str(args.live_stream_file.resolve())
    return subprocess.run([str(BLENDER), "--factory-startup", "--python",
                           str(ROOT / "scripts/blender_world.py")], cwd=ROOT, env=env,
                          check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
