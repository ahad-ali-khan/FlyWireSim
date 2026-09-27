"""Held-out catch evaluation, separate from stochastic training rewards."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .baselines import deterministic_pursuit_baseline, random_action
from .cognitive import CognitiveActorCritic
from .env import HuntEnv, PHYSICS_VERSION
from .train_cognitive_sac import legacy_moth_observation, bat_policy_observation
from .training_runtime import atomic_json


def evaluate(bat, moth, config, seeds=100, seed_start=50000, policy="trained", mode="mix",
             observation_frame="world", recorded_frames=None, layout_seed=None):
    outcomes = []
    # Evaluation must not advance the trainer's exploration RNG.
    with torch.random.fork_rng(), torch.no_grad():
        for seed in range(seed_start, seed_start + seeds):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            bat_rng = np.random.default_rng(seed + 987654)
            opponent = rng.choice(["learned", "random", "stationary"], p=[.5, .3, .2]) if mode == "mix" else mode
            # Held-out positions AND obstacle layouts; same seeds for all arms.
            env = HuntEnv(config=config | {"obstacle_seed": seed if layout_seed is None else layout_seed,
                                          "stationary_moth": opponent == "stationary"})
            env.reset(seed=seed)
            bat_hidden, moth_hidden = bat.initial_hidden(), moth.initial_hidden()
            reward, aligned, sonar = 0., 0, 0
            while True:
                obs = env.observe("bat")
                if policy in ("trained", "untrained"):
                    policy_obs = bat_policy_observation(obs, observation_frame)
                    output = bat.forward_sequence(torch.tensor(policy_obs).reshape(1, 1, -1), bat_hidden)
                    action = output.mean[0, 0].tanh().numpy()
                    bat_hidden = output.hidden
                elif policy == "random":
                    action = random_action(bat_rng)
                else:
                    action = deterministic_pursuit_baseline(env.bat.position, env.moth.position,
                        env.bat.velocity, env.bat.yaw, env.bat.pitch,
                        max_yaw_delta=env.cfg["max_yaw_delta"], max_pitch_delta=env.cfg["max_pitch_delta"],
                        bat_accel=env.cfg["bat_accel"], dt=env.cfg["dt"],
                        drag=env.cfg["drag"], target_speed=env.cfg["bat_max_speed"])
                if opponent == "learned":
                    moth_obs = legacy_moth_observation(env.observe("moth"))
                    out = moth.forward_sequence(torch.tensor(moth_obs).reshape(1, 1, -1), moth_hidden)
                    moth_hidden = out.hidden
                    moth_action = torch.distributions.Normal(out.mean[0, 0], moth.log_standard_deviation.exp()).sample().clamp(-1, 1).numpy()
                elif opponent == "random":
                    moth_action = random_action(rng)
                else:
                    moth_action = np.array([0, 0, -1], dtype=np.float32)
                _, r, done, _, info = env.step_with_actions({"bat": action, "moth": moth_action})
                if recorded_frames is not None:
                    recorded_frames.append({
                        "playback_kind": "evaluation", "policy_label": "SAC evaluation / frozen policies",
                        "phase": "SAC-evaluation", "episode": seed - seed_start + 1, "tick": env.tick,
                        "caught": info["caught"], "timeout": info["timeout"],
                        "jaw_distance": info["jaw_distance"], "swept_jaw_distance": info["swept_jaw_distance"],
                        "jaw_contact_fraction": info["jaw_contact_fraction"],
                        "jaw_contact_position": info["jaw_contact_position"],
                        "bat_sonar_hit": info["bat_alignment"] is not None,
                        "bat_alignment": info["bat_alignment"],
                        "bat_alignment_reward": info["bat_alignment_reward"],
                        "bat_aligned_thrust_reward": 0.,
                        "sonar_occluded": info["sonar_occluded"], "vision_occluded": info["vision_occluded"],
                        "bat_action": action.tolist(), "moth_opponent_mode": str(opponent),
                        "bat": dict(position=info["bat_position"], velocity=info["bat_velocity"],
                                    sonar_radius=env.cfg["sonar_range"], jaw_position=info["mouth_position"]),
                        "moth": dict(position=info["moth_position"], velocity=info["moth_velocity"],
                                     flame_position=info["flame_position"]),
                        "environment": {"obstacles": [dict(x=float(o["xy"][0]), y=float(o["xy"][1]),
                            angle=float(np.arctan2(o["branch"][1], o["branch"][0]))) for o in env.obstacles]},
                    })
                reward += r
                if info["bat_alignment"] is not None:
                    sonar += 1
                    aligned += info["bat_alignment"] >= .4
                if done:
                    break
            outcomes.append(dict(seed=seed, opponent=str(opponent), caught=info["caught"],
                ticks=env.tick, reward=reward, aligned_ticks=aligned, sonar_ticks=sonar))
    by_opponent = {}
    for name in ("learned", "random", "stationary"):
        rows = [e for e in outcomes if e["opponent"] == name]
        by_opponent[name] = dict(episodes=len(rows),
            catch_rate=float(np.mean([e["caught"] for e in rows])) if rows else None)
    return dict(policy="deterministic_pursuit_baseline" if policy == "pursuit" else policy,
        evaluation="deterministic bat, stochastic frozen moth, held-out layouts",
        mode=mode, seeds=seeds, seed_start=seed_start,
        catch_rate=float(np.mean([e["caught"] for e in outcomes])),
        mean_reward=float(np.mean([e["reward"] for e in outcomes])),
        by_opponent=by_opponent,
        episodes=outcomes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=50000)
    parser.add_argument("--policies", nargs="+", choices=("trained", "untrained", "random", "pursuit"), default=["trained"])
    parser.add_argument("--modes", nargs="+", choices=("mix", "stationary", "random", "learned"), default=["mix", "stationary"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("physics_version") != PHYSICS_VERSION:
        parser.error("checkpoint lacks matching physics metadata; use the legacy diagnostic explicitly")
    bat, moth = CognitiveActorCritic("bat", 19).eval(), CognitiveActorCritic("moth", 23).eval()
    moth.load_state_dict(state["moth"])
    results = []
    for policy in args.policies:
        if policy == "untrained":
            torch.manual_seed(state["seed"])
            bat = CognitiveActorCritic("bat", 19).eval()
        else:
            bat.load_state_dict(state["bat"])
        for mode in args.modes:
            result = evaluate(bat, moth, state["environment_config"], args.seeds, args.seed_start, policy, mode,
                              state.get("observation_frame", "world"))
            results.append(result)
            print(f"{policy} / {mode}: {result['catch_rate']:.1%}, reward {result['mean_reward']:+.3f}", flush=True)
            atomic_json(args.output, dict(checkpoint=str(args.checkpoint), episode=state["next_episode"],
                physics_version=PHYSICS_VERSION, observation_frame=state.get("observation_frame", "world"),
                environment_config=state["environment_config"], results=results))


if __name__ == "__main__":
    main()
