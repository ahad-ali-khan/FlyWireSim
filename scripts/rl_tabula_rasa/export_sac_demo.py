"""Export consecutive real learned-policy hunts for buffered Blender playback."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from .cognitive import CognitiveActorCritic
from .evaluate_cognitive_sac import evaluate
from .train_sac_control import FlatEvaluationAdapter
from .training_runtime import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--moth-opponent-checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True, help="new .jsonl frame stream")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output path; recordings are not overwritten")
    torch.set_num_threads(1)
    run = json.loads(args.run.read_text())
    model = SAC.load(args.run.with_suffix(".zip"), device="cpu")
    moth = CognitiveActorCritic("moth", 23).eval()
    moth.load_state_dict(torch.load(args.moth_opponent_checkpoint, map_location="cpu", weights_only=False)["models"]["moth"])
    frames = []
    result = evaluate(FlatEvaluationAdapter(model), moth, run["environment_config"],
        seeds=args.episodes, seed_start=80000, observation_frame="body",
        recorded_frames=frames, layout_seed=run["seed"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        for frame in frames:
            handle.write(json.dumps(frame, separators=(",", ":")) + "\n")
    atomic_json(args.output.with_suffix(".state.json"), frames[0])
    atomic_json(args.output.with_suffix(".manifest.json"), dict(run=str(args.run),
        label="flat SAC, deterministic bat / frozen moth mix; recorded evaluation, not live learning",
        source="30 consecutive seeds by default; no cherry-picked catches",
        obstacle_layout="fixed training layout for the demo; benchmarks use held-out layouts", result=result))
    episodes = run["episodes"]
    atomic_json(args.output.with_suffix(".curve.json"), {"curve": [
        dict(step=i+1, catch_rate_last_100=float(np.mean([e["caught"] for e in episodes[max(0, i-99):i+1]])))
        for i in range(len(episodes))]})
    print(f"Saved {len(frames)} frames, {sum(e['caught'] for e in result['episodes'])}/{args.episodes} catches: {args.output}")


if __name__ == "__main__":
    main()
