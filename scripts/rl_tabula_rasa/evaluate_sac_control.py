"""Evaluate a saved flat SAC control on untouched, hard-start test seeds."""
import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import SAC

from .cognitive import CognitiveActorCritic
from .evaluate_cognitive_sac import evaluate
from .train_sac_control import FlatEvaluationAdapter
from .training_runtime import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="control run JSON, paired with its .zip")
    parser.add_argument("--moth-opponent-checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=60000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run = json.loads(args.run.read_text())
    model = SAC.load(args.run.with_suffix(".zip"), device="cpu")
    moth = CognitiveActorCritic("moth", 23).eval()
    moth.load_state_dict(torch.load(args.moth_opponent_checkpoint, map_location="cpu", weights_only=False)["models"]["moth"])
    results = []
    for mode in ("mix", "stationary"):
        result = evaluate(FlatEvaluationAdapter(model), moth, run["environment_config"],
            seeds=args.seeds, seed_start=args.seed_start, mode=mode, observation_frame="body")
        results.append(result)
        print(f"flat SAC seed {run['seed']} / {mode}: {result['catch_rate']:.1%}", flush=True)
        atomic_json(args.output, dict(algorithm=run["algorithm"], seed=run["seed"],
            run=str(args.run), curriculum_steps=run.get("curriculum_steps", 0), results=results))


if __name__ == "__main__":
    main()
