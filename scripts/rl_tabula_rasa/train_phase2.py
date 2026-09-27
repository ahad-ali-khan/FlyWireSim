"""Compatibility entry point; episode-synchronous self-play is the default.

Pass --legacy-alternating to explicitly run the earlier alternating-block method.
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .novelty import (NOVELTY_THRESHOLD, NOVELTY_WARNING_RATE, NOVELTY_WEIGHT,
                      REPERTOIRE_SIZE, NoveltyEnv)
from .train import LOG_INTERVAL, MOVING_WINDOW, RESULTS
from .training_runtime import (DATA, TRAINING_MAX_THREADS, TickPacer, TrainingRuntime,
                               atomic_json, latest_full_checkpoint, load_full_policies,
                               restore_rng)


STOP_REQUESTED = threading.Event()


class Metrics(BaseCallback):
    def __init__(self, role, runtime=None, pacer=None):
        super().__init__()
        self.role = role
        self.runtime = runtime
        self.pacer = pacer
        self.episodes = []
        self.next_log = LOG_INTERVAL

    def _on_training_start(self):
        self.next_log = ((self.num_timesteps // LOG_INTERVAL) + 1) * LOG_INTERVAL

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if self.runtime:
                self.runtime.write_live_state(self.role, info)
            if "episode" in info:
                self.episodes.append({"step": self.num_timesteps,
                                      "caught": bool(info["caught"]),
                                      "hunt_length": int(info["hunt_length"]),
                                      "novel_behavior": bool(info["novel_behavior"][self.role]),
                                      "novelty_distance": float(info["behavior_metrics"][self.role]["novelty_distance"]),
                                      "nearest_repertoire_similarity": info["behavior_metrics"][self.role]["nearest_repertoire_similarity"],
                                      "legacy_reference_similarity": float(info["legacy_reference_similarity"][self.role])})
        if self.num_timesteps >= self.next_log:
            recent = self.episodes[-MOVING_WINDOW:]
            print(f"phase2 {self.role} step {self.num_timesteps}: "
                  f"catch {np.mean([e['caught'] for e in recent]) if recent else 0:.3f}, "
                  f"novel behavior rate {np.mean([e['novel_behavior'] for e in recent]) if recent else 0:.3f}",
                  flush=True)
            if len(recent) >= 20 and np.mean([e["novel_behavior"] for e in recent]) >= NOVELTY_WARNING_RATE:
                print("WARNING: novel behavior rate is >=95%; review repertoire threshold/behavior diversity.",
                      flush=True)
            self.next_log += LOG_INTERVAL
        if self.runtime:
            self.runtime.maybe_checkpoint()
        if STOP_REQUESTED.is_set():
            if self.runtime:
                self.runtime.maybe_checkpoint(force=True, reason="shutdown")
            return False
        return self.pacer.pace(STOP_REQUESTED.is_set) if self.pacer else True


def summary(episodes):
    if not episodes:
        return {"episodes": 0}
    last = episodes[-MOVING_WINDOW:]
    return {"episodes": len(episodes),
            "catch_rate_last_100": float(np.mean([e["caught"] for e in last])),
            "novel_behavior_rate_last_100": float(np.mean([e["novel_behavior"] for e in last])),
            "legacy_reference_similarity_last_100": float(np.mean(
                [e["legacy_reference_similarity"] for e in last])),
            "average_hunt_length_last_100": float(np.mean([e["hunt_length"] for e in last]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-per-role", type=int, default=50_000)
    parser.add_argument("--block-steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int,
                        help="random seed; on resume defaults to the seed recorded in the saved artifact")
    parser.add_argument("--output", type=Path,
                        help="new JSON output path; defaults to a UTC-timestamped results file")
    parser.add_argument("--resume-from", type=Path,
                        help="continue both PPO policies and repertoires from a saved Phase 2 JSON artifact")
    parser.add_argument("--continuous", action="store_true",
                        help="train indefinitely; startup automatically resumes the newest full checkpoint")
    parser.add_argument("--no-pacing", action="store_true",
                        help="disable real-time and battery pacing for short tests")
    args = parser.parse_args()
    if args.steps_per_role < args.block_steps or args.block_steps < 1024:
        parser.error("steps-per-role must be >= block-steps >= 1024")
    full_path = latest_full_checkpoint() if not args.resume_from else None
    legacy_phase2_path = None
    if not args.resume_from and not full_path:
        candidates = sorted(RESULTS.glob("phase2_coevolution_*.json"))
        legacy_phase2_path = next((path for path in reversed(candidates)
                                   if "checkpoints" in json.loads(path.read_text())
                                   and "behavioral_repertoires" in json.loads(path.read_text())), None)
    prior_path = args.resume_from or legacy_phase2_path
    prior = json.loads(prior_path.read_text()) if prior_path else None
    full_prior = json.loads(full_path.read_text()) if full_path else None
    restored = prior or full_prior
    if not restored:
        gate_path = RESULTS / "phase0_learning_curve.json"
        gate_result = json.loads(gate_path.read_text())
        if not gate_result["gate"]["passed"]:
            parser.error("Phase 0 learning gate has not passed")
        if not gate_result.get("collision_model", "").startswith("jaw offset measured"):
            parser.error("Phase 0 checkpoint uses the obsolete body-center contact model; retrain with jaw calibration")
    seed = args.seed if args.seed is not None else int(restored.get("seed", 23)) if restored else 23
    bat_env = Monitor(NoveltyEnv("bat"))
    moth_env = Monitor(NoveltyEnv("moth"))
    bat_env.reset(seed=seed)
    moth_env.reset(seed=seed + 1)
    envs = {"bat": bat_env, "moth": moth_env}
    if full_prior:
        models, _ = load_full_policies(full_path, envs)
        bat, moth = models["bat"], models["moth"]
        bat_env.unwrapped.load_novelty_state(full_prior.get("behavioral_repertoires", {}).get("bat", {}))
        moth_env.unwrapped.load_novelty_state(full_prior.get("behavioral_repertoires", {}).get("moth", {}))
        print(f"Resumed step {full_prior['steps_per_role']} from full checkpoint {full_path.name}.", flush=True)
    elif prior:
        checkpoint_paths = prior.get("checkpoints", {})
        resume_bat = Path(checkpoint_paths.get("bat", ""))
        resume_moth = Path(checkpoint_paths.get("moth", ""))
        if not resume_bat.is_file() or not resume_moth.is_file():
            parser.error("resume artifact must point to both saved bat and moth checkpoints")
        bat = PPO.load(resume_bat, env=bat_env, device="cpu")
        moth = PPO.load(resume_moth, env=moth_env, device="cpu")
        bat_env.unwrapped.load_novelty_state(prior.get("behavioral_repertoires", {}).get("bat_env", {}))
        moth_env.unwrapped.load_novelty_state(prior.get("behavioral_repertoires", {}).get("moth_env", {}))
        if legacy_phase2_path:
            print(f"No full checkpoint yet; continuing prior Phase 2 progress from {legacy_phase2_path.name}.",
                  flush=True)
    else:
        bat_path, moth_path = RESULTS / "phase0_bat_ppo.zip", RESULTS / "phase0_moth_ppo.zip"
        if not bat_path.exists() or not moth_path.exists():
            parser.error("Phase 0 PPO checkpoints missing; train and pass its gate first")
        bat = PPO.load(bat_path, env=bat_env, device="cpu")
        moth = PPO.load(moth_path, env=moth_env, device="cpu")
    bat_env.unwrapped.opponent = moth
    moth_env.unwrapped.opponent = bat
    metrics = {"bat": Metrics("bat"), "moth": Metrics("moth")}
    if restored:
        for role, callback in metrics.items():
            callback.episodes = list(restored.get("roles", {}).get(role, {}).get("episodes", []))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or RESULTS / f"phase2_coevolution_{stamp}.json"
    bat_checkpoint = RESULTS / f"phase2_bat_ppo_{stamp}"
    moth_checkpoint = RESULTS / f"phase2_moth_ppo_{stamp}"
    import torch
    torch.set_num_threads(TRAINING_MAX_THREADS)
    pacer = TickPacer(enabled=not args.no_pacing)
    epochs = list(restored.get("epochs", [])) if restored else []
    starting_steps = int(restored.get("steps_per_role", 0)) if restored else 0
    runtime = TrainingRuntime({"bat": bat, "moth": moth}, envs, metrics, seed,
                              epochs, output_path, starting_steps)
    for role, callback in metrics.items():
        callback.runtime = runtime
        callback.pacer = pacer

    def request_stop(_signum, _frame):
        STOP_REQUESTED.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    completed = 0
    while args.continuous or completed < args.steps_per_role:
        if STOP_REQUESTED.is_set():
            break
        block = args.block_steps if args.continuous else min(args.block_steps, args.steps_per_role - completed)
        epoch_starts = {role: len(callback.episodes) for role, callback in metrics.items()}
        for role, model in (("bat", bat), ("moth", moth)):
            model.learn(total_timesteps=block, reset_num_timesteps=False, callback=metrics[role])
            model.save(bat_checkpoint if role == "bat" else moth_checkpoint)
            if STOP_REQUESTED.is_set():
                break
        if STOP_REQUESTED.is_set():
            break
        completed += block
        runtime.steps_per_role = starting_steps + completed
        epoch_metrics = {}
        for role, callback in metrics.items():
            block_episodes = callback.episodes[epoch_starts[role]:]
            novel_rate = (float(np.mean([episode["novel_behavior"] for episode in block_episodes]))
                          if block_episodes else None)
            epoch_metrics[role] = {"episodes": len(block_episodes),
                                   "novel_behavior_rate": novel_rate}
            if novel_rate is not None and len(block_episodes) >= 20 and novel_rate >= NOVELTY_WARNING_RATE:
                print(f"WARNING: {role} novel behavior rate is {novel_rate:.3f} this epoch; "
                      "review repertoire threshold/behavior diversity.", flush=True)
        epochs.append({"steps_per_role": starting_steps + completed, "roles": epoch_metrics})
        result = {"phase": 2, "seed": seed, "steps_per_role": starting_steps + completed,
                  "additional_training_steps": completed,
                  "method": "alternating PPO blocks against current opponent; no discrete action selection",
                  "epochs": epochs,
                  "novelty": {"repertoire_size": REPERTOIRE_SIZE, "weight": NOVELTY_WEIGHT,
                              "similarity_threshold": NOVELTY_THRESHOLD,
                              "reported_rate": "episodes added to the per-role behavioral repertoire / completed episodes",
                              "legacy_reference_similarity": "diagnostic only; excluded from benchmark novelty"},
                  "roles": {role: {"summary": summary(callback.episodes),
                                   "episodes": callback.episodes}
                            for role, callback in metrics.items()},
                  "behavioral_repertoires": {"bat_env": bat_env.unwrapped.novelty_state(),
                                             "moth_env": moth_env.unwrapped.novelty_state()},
                  "checkpoints": {"bat": str(bat_checkpoint.with_suffix(".zip")),
                                  "moth": str(moth_checkpoint.with_suffix(".zip"))}}
        atomic_json(output_path, result)
        if STOP_REQUESTED.is_set():
            break
    final_checkpoint = runtime.save_full_checkpoint("shutdown" if STOP_REQUESTED.is_set() else "completion")
    print(f"Full checkpoint saved: {final_checkpoint}", flush=True)
    print(f"Phase 2 PPO checkpoints and metrics saved to {output_path}; legacy Blender viewer remains separate.",
          flush=True)


if __name__ == "__main__":
    import sys
    if "--legacy-alternating" in sys.argv:
        sys.argv.remove("--legacy-alternating")
        main()
    else:
        from .train_selfplay import main as selfplay_main
        selfplay_main()
