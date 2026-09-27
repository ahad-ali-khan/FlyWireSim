"""Run reproducible PPO, non-learning, Q-learning, and connectome-seed evaluations."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coevolution import AdaptivePolicy
from rl_tabula_rasa.baselines import deterministic_pursuit_baseline, random_action
from rl_tabula_rasa.connectome import apply_seed, load_seed
from rl_tabula_rasa.env import CONFIG, HuntEnv
from rl_tabula_rasa.novelty import (NOVELTY_THRESHOLD, REPERTOIRE_SIZE,
                                    BehavioralRepertoire, NoveltyEnv, action_signature)
from rl_tabula_rasa.train import PPO_CONFIG, RESULTS
from rl_tabula_rasa.train_selfplay import train_joint_episode
from rl_tabula_rasa.training_runtime import TRAINING_MAX_THREADS, atomic_json


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("random_action_baseline", "original_discrete_primitive_q_learning_baseline",
             "phase0_flat_mlp_frozen_baseline", "deterministic_pursuit_baseline")


def _stats(episodes):
    if not episodes:
        return {"episodes": 0, "catch_rate": None, "moth_survival_rate": None,
                "novel_behavior_rate": None, "average_hunt_length": None}
    caught = np.asarray([ep["caught"] for ep in episodes], dtype=float)
    return {"episodes": len(episodes), "catch_rate": float(caught.mean()),
            "moth_survival_rate": float(1 - caught.mean()),
            "novel_behavior_rate": float(np.mean([ep["novel_behavior"] for ep in episodes])),
            "average_hunt_length": float(np.mean([ep["hunt_length"] for ep in episodes]))}


def _curve(episodes, interval=1000):
    points, ticks = [], 0
    for index, ep in enumerate(episodes, 1):
        ticks += ep["hunt_length"]
        if ticks >= interval and (not points or ticks - points[-1]["step"] >= interval):
            window = episodes[max(0, index - 100):index]
            points.append({"step": ticks, "catch_rate_last_100": float(np.mean([x["caught"] for x in window]))})
    return points


def _aggregate(runs):
    fields = ("catch_rate", "moth_survival_rate", "novel_behavior_rate", "average_hunt_length")
    result = {}
    for label, by_seed in runs.items():
        values = [item for item in by_seed.values() if "episodes" in item]
        groups = {"full_run": values, "final_20_percent": [
            {**run, "episodes": run["episodes"][int(len(run["episodes"]) * 0.8):]}
            for run in values]}
        result[label] = {}
        for portion, group in groups.items():
            result[label][portion] = {}
            for field in fields:
                samples = [_stats(run["episodes"])[field] for run in group]
                samples = [value for value in samples if value is not None]
                result[label][portion][field] = {
                    "mean": float(np.mean(samples)) if samples else None,
                    "std": float(np.std(samples, ddof=1)) if len(samples) > 1 else (0.0 if samples else None),
                    "n": len(samples)}
    return result


def _features(env, role, previous_distance):
    distance = float(np.linalg.norm(env.bat.position - env.moth.position))
    body = env.bat if role == "bat" else env.moth
    wall = min(CONFIG["arena_x"] - abs(float(body.position[0])),
               CONFIG["arena_y"] - abs(float(body.position[1])),
               float(body.position[2]) - CONFIG["floor_z"],
               CONFIG["ceiling_z"] - float(body.position[2]))
    approach = np.clip((previous_distance - distance) * 8.0, -1.0, 1.0)
    if role == "bat":
        return (1.0, np.clip(1 - distance / 5.0, 0, 1),
                np.clip(np.linalg.norm(env.moth.velocity) * 8.0, 0, 1),
                np.clip(1 - wall / 1.4, 0, 1), 0.5)
    flame_distance = float(np.linalg.norm(env.moth.position - env.flame))
    return (1.0, np.clip(1 - distance / 3.4, 0, 1),
            np.clip((approach + 1) / 2, 0, 1), np.clip(1 - wall / 1.4, 0, 1),
            np.clip(1 - flame_distance / 3.0, 0, 1))


def _steer(position, velocity, yaw, pitch, target, species):
    return deterministic_pursuit_baseline(position, target, velocity, yaw, pitch,
        max_yaw_delta=CONFIG["max_yaw_delta"], max_pitch_delta=CONFIG["max_pitch_delta"],
        bat_accel=CONFIG[f"{species}_accel"], drag=CONFIG["drag"], dt=CONFIG["dt"],
        target_speed=CONFIG[f"{species}_max_speed"])


def _primitive_action(env, role, tactic, tick, rng):
    if role == "bat":
        target = env.moth.position.copy()
        if tactic == "LEAD":
            target += env.moth.velocity * 8.0
        elif tactic == "SWEEP":
            target += np.asarray((0.35 * math.sin(tick * 0.08), 0.35 * math.cos(tick * 0.08), 0.0))
        body = env.bat
        return _steer(body.position, body.velocity, body.yaw, body.pitch, target, "bat")

    body = env.moth
    away = body.position - env.bat.position
    away_norm = away / max(1e-8, float(np.linalg.norm(away)))
    lateral = np.asarray((-away_norm[1], away_norm[0], 0.0), dtype=np.float32)
    light = env.flame - body.position
    light /= max(1e-8, float(np.linalg.norm(light)))
    distance = float(np.linalg.norm(away))
    if tactic == "DODGE":
        desired = away_norm * 0.6 + lateral * (2.4 if distance < 2.5 else 0.4)
    elif tactic == "FLANK":
        desired = light * 0.55 + lateral * 1.8 + away_norm * 0.25
    elif tactic == "LATE":
        desired = light if distance > 1.3 else lateral * 2.6 + away_norm * 0.9
    else:  # FEINT retains an arbitrary, seeded side sign but does not script escapes.
        sign = -1.0 if (tick // 12) % 2 else 1.0
        desired = light * 0.7 + lateral * sign * 0.9
        if distance < 1.55:
            desired = lateral * -sign * 2.6 + away_norm * 0.6
    target = body.position + desired
    return _steer(body.position, body.velocity, body.yaw, body.pitch, target, "moth")


def _q_controllers(env, tick, previous_distance, policies, rng):
    actions = {}
    features = {role: _features(env, role, previous_distance) for role in policies}
    primitives = {"bat": ("DIRECT", "LEAD", "SWEEP"),
                  "moth": ("DODGE", "FLANK", "LATE", "FEINT")}
    for role, policy in policies.items():
        if policy.current is None:
            policy.choose(features[role])
        tactics = policy.current.split("+")
        actions[role] = np.mean([_primitive_action(env, role, tactic, tick, rng)
                                 for tactic in tactics], axis=0).astype(np.float32)
    return actions


def _new_model(role, role_env, seed, *, init_from_phase0=True):
    if init_from_phase0:
        return PPO.load(RESULTS / f"phase0_{role}_ppo.zip", env=role_env, device="cpu")
    return PPO("MlpPolicy", role_env, seed=seed, verbose=0, device="cpu",
               n_steps=PPO_CONFIG["n_steps"], batch_size=PPO_CONFIG["batch_size"],
               n_epochs=PPO_CONFIG["n_epochs"], learning_rate=PPO_CONFIG["learning_rate"],
               ent_coef=PPO_CONFIG["entropy_coefficient"])


def _trained_run(seed, budget, *, seeded_moth=False, fresh_moth=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    joint = NoveltyEnv("bat")
    envs = {"bat": Monitor(NoveltyEnv("bat")), "moth": Monitor(NoveltyEnv("moth"))}
    bat = _new_model("bat", envs["bat"], seed)
    moth = _new_model("moth", envs["moth"], seed + 1, init_from_phase0=not fresh_moth)
    if seeded_moth:
        weights, _ = load_seed(RESULTS / "connectome_seed.json")
        apply_seed(moth, weights)
    models = {"bat": bat, "moth": moth}
    for model in models.values():
        model._setup_learn(total_timesteps=budget, reset_num_timesteps=False)
    total_budget = max(model._total_timesteps for model in models.values())
    episodes, steps, next_curve = [], 0, 1000
    while steps < budget:
        length, info = train_joint_episode(models, joint, envs, seed + len(episodes),
                                           total_budget)
        steps += length
        metrics = info.get("behavior_metrics", {}).get("bat", {})
        episodes.append({"caught": bool(info["caught"]), "hunt_length": length,
                         "novel_behavior": bool(metrics.get("novel_behavior", False))})
        episodes[-1]["selfplay_step"] = steps
        episodes[-1]["bat_policy_steps"] = int(bat.num_timesteps)
        episodes[-1]["moth_policy_steps"] = int(moth.num_timesteps)
        episodes[-1]["selfplay_log"] = f"self-play: bat step {steps}, moth step {steps}"
        if steps >= next_curve:
            recent = episodes[-100:]
            print(f"episode_synchronous_ppo seed {seed} step {steps}: "
                  f"catch {np.mean([ep['caught'] for ep in recent]):.3f}; "
                  f"novel {np.mean([ep['novel_behavior'] for ep in recent]):.3f}", flush=True)
            next_curve += 1000
    return {"steps": steps, "episodes": episodes, "curve": _curve(episodes),
            "summary": _stats(episodes), "connectome_seeded_moth": seeded_moth,
            "CONNECTOME_SEEDED": seeded_moth,
            "moth_initialization": "FlyWire optic-pathway seed" if seeded_moth else
                                  "unseeded PPO initialization" if fresh_moth else "Phase 0 PPO"}


def _baseline_run(label, seed, budget):
    random.seed(seed)
    rng = np.random.default_rng(seed)
    env = HuntEnv("bat")
    frozen = None
    if label == "phase0_flat_mlp_frozen_baseline":
        frozen = {role: PPO.load(RESULTS / f"phase0_{role}_ppo.zip", device="cpu")
                  for role in ("bat", "moth")}
    policies = None
    if label == "original_discrete_primitive_q_learning_baseline":
        policies = {"bat": AdaptivePolicy(("DIRECT", "LEAD", "SWEEP"), rng=random.Random(seed)),
                    "moth": AdaptivePolicy(("DODGE", "FLANK", "LATE", "FEINT"), rng=random.Random(seed + 1))}
    episodes, steps, index, next_curve = [], 0, 0, 1000
    repertoire = BehavioralRepertoire()
    while steps < budget:
        env.reset(seed=seed + index)
        index += 1
        actions_seen = {"bat": [], "moth": []}
        old_distance = float(np.linalg.norm(env.bat.position - env.moth.position))
        for tick in range(env.cfg["max_ticks"]):
            if label == "random_action_baseline":
                actions = {role: random_action(rng) for role in ("bat", "moth")}
            elif label == "deterministic_pursuit_baseline":
                actions = {"bat": deterministic_pursuit_baseline(
                    env.bat.position, env.moth.position, env.bat.velocity, env.bat.yaw,
                    env.bat.pitch, max_yaw_delta=env.cfg["max_yaw_delta"],
                    max_pitch_delta=env.cfg["max_pitch_delta"], bat_accel=env.cfg["bat_accel"],
                    drag=env.cfg["drag"], dt=env.cfg["dt"], target_speed=env.cfg["bat_max_speed"]),
                    "moth": random_action(rng)}
            elif frozen:
                actions = {role: frozen[role].predict(env.observe(role), deterministic=True)[0]
                           for role in frozen}
            else:
                actions = _q_controllers(env, tick, old_distance, policies, rng)
            _, _, done, _, info = env.step_with_actions(actions)
            for role in actions_seen:
                actions_seen[role].append(actions[role])
            steps += 1
            if policies:
                distance_now = float(np.linalg.norm(env.bat.position - env.moth.position))
                for role, policy in policies.items():
                    policy.step(_features(env, role, old_distance), info[f"{role}_reward"],
                                terminal=done)
                old_distance = distance_now
            if done:
                novelty = {role: repertoire.classify_and_add(
                    role, action_signature(actions_seen[role]))[0] for role in actions_seen}
                episodes.append({"caught": bool(info["caught"]), "hunt_length": tick + 1,
                                 "novel_behavior": bool(novelty["bat"]),
                                 "novel_behavior_by_role": novelty})
                break
        if steps >= next_curve:
            recent = episodes[-100:]
            print(f"{label} seed {seed} step {steps}: "
                  f"catch {np.mean([ep['caught'] for ep in recent]):.3f}", flush=True)
            next_curve += 1000
    return {"steps": steps, "episodes": episodes, "curve": _curve(episodes),
            "summary": _stats(episodes), "baseline": True,
            "baseline_name": label,
            "note": "Reference behavior only; not evidence of learning."}


def _save(path, payload):
    atomic_json(path, payload)


def _ensure_episode_audit(runs):
    phase0_steps = {role: PPO.load(RESULTS / f"phase0_{role}_ppo.zip",
                                   device="cpu").num_timesteps
                    for role in ("bat", "moth")}
    for label, by_seed in runs.items():
        if label != "episode_synchronous_ppo" and not label.startswith("connectome_"):
            continue
        for run in by_seed.values():
            steps = 0
            moth_start = 0 if label.startswith("connectome_") else phase0_steps["moth"]
            for episode in run.get("episodes", []):
                steps += int(episode["hunt_length"])
                episode.setdefault("selfplay_step", steps)
                episode.setdefault("bat_policy_steps", phase0_steps["bat"] + steps)
                episode.setdefault("moth_policy_steps", moth_start + steps)
                episode.setdefault("selfplay_log", f"self-play: bat step {steps}, moth step {steps}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps-per-seed", type=int,
                        help="environment ticks per run; defaults to the gated Phase 0 training length")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    gate = json.loads((RESULTS / "phase0_learning_curve.json").read_text())
    if not gate["gate"]["passed"]:
        parser.error("Phase 0 learning gate has not passed")
    budget = args.steps_per_seed or int(gate["target_training_steps"])
    torch.set_num_threads(TRAINING_MAX_THREADS)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or RESULTS / f"benchmark_multiseed_{stamp}.json"
    if output.resolve() == (RESULTS / "benchmark.json").resolve():
        parser.error("refusing to overwrite the immutable original results/benchmark.json")
    payload = json.loads(output.read_text()) if args.resume and output.exists() else {
        "format": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "steps_per_seed": budget, "seed_count": args.seeds,
        "baseline_labels": list(BASELINES),
        "novelty_metric": {"name": "corrected novel behavior rate",
            "definition": "episodes added to the per-role K-entry repertoire / completed episodes",
            "distance": "cosine distance over fixed-length resampled action sequences",
            "repertoire_size": REPERTOIRE_SIZE, "similarity_threshold": NOVELTY_THRESHOLD},
        "collision_geometry": {"jaw_offset": CONFIG["jaw_forward_offset"],
                               "contact_radius": CONFIG["jaw_contact_radius"],
                               "physics": "shared HuntEnv 3D dynamics and mouth-distance terminal collision"},
        "seeded_comparison": "same Phase 0 bat checkpoint; unseeded vs FlyWire-seeded fresh moth PPO, co-trained per episode",
        "runs": {}, "status": "running"}
    payload.setdefault("baseline_labels", list(BASELINES))
    payload.setdefault("novelty_metric", {"name": "corrected novel behavior rate",
        "definition": "episodes added to the per-role K-entry repertoire / completed episodes",
        "distance": "cosine distance over fixed-length resampled action sequences",
        "repertoire_size": REPERTOIRE_SIZE, "similarity_threshold": NOVELTY_THRESHOLD})
    payload.setdefault("collision_geometry", {"jaw_offset": CONFIG["jaw_forward_offset"],
        "contact_radius": CONFIG["jaw_contact_radius"],
        "physics": "shared HuntEnv 3D dynamics and mouth-distance terminal collision"})
    seed_ids = list(range(args.seeds))

    def persist():
        payload["aggregates"] = _aggregate(payload["runs"])
        _save(output, payload)

    for seed in seed_ids:
        label = "episode_synchronous_ppo"
        if str(seed) not in payload["runs"].setdefault(label, {}):
            print(f"Running PPO self-play seed {seed}/{args.seeds - 1} for {budget} ticks.", flush=True)
            payload["runs"][label][str(seed)] = _trained_run(seed + 23, budget)
            persist()
        for label in BASELINES:
            if str(seed) not in payload["runs"].setdefault(label, {}):
                print(f"Running {label} seed {seed}/{args.seeds - 1} for {budget} ticks.", flush=True)
                payload["runs"][label][str(seed)] = _baseline_run(label, seed + 23, budget)
                persist()
        for seeded, name in ((False, "connectome_unseeded_moth"),
                             (True, "connectome_seeded_moth")):
            if str(seed) not in payload["runs"].setdefault(name, {}):
                print(f"Running {name} seed {seed}/{args.seeds - 1} for {budget} ticks.", flush=True)
                payload["runs"][name][str(seed)] = _trained_run(
                    seed + 23, budget, seeded_moth=seeded, fresh_moth=True)
                persist()
    payload["status"] = "complete"
    _ensure_episode_audit(payload["runs"])
    payload["connectome_seed_metadata"] = load_seed(
        RESULTS / "connectome_seed.json")[1]
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    persist()
    print(f"Saved multi-seed benchmark: {output}", flush=True)


if __name__ == "__main__":
    main()
