"""Run with python -m scripts.rl_tabula_rasa.test_sac (CPU, no training run)."""
import copy
import numpy as np
import torch

from .cognitive import CognitiveActorCritic
from .env import HuntEnv, REFERENCE_DT, _swept_contact
from .train_cognitive_sac import (EpisodeReplayBuffer, TwinCritic, _sac_update,
                                  bat_policy_observation)


def checks():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    rng = np.random.default_rng(2)
    # Consistent unit conversion preserves trajectories for BOTH species and
    # boundary repulsion, including nonzero initial velocity.
    legacy = HuntEnv(config=dict(dt=1, bat_accel=.22, moth_accel=.15,
        bat_max_speed=.42, moth_max_speed=.28, boundary_k=.012, max_repulsion=.35))
    si = HuntEnv()
    for env in (legacy, si):
        env.reset(seed=3)
    for _ in range(100):
        for role in ("bat", "moth"):
            action = rng.uniform(-1, 1, 3)
            legacy._move(getattr(legacy, role), action, role)
            si._move(getattr(si, role), action, role)
            assert np.allclose(getattr(si, role).position, getattr(legacy, role).position, atol=5e-5)
            assert np.allclose(getattr(si, role).velocity * REFERENCE_DT,
                               getattr(legacy, role).velocity, atol=5e-5)

    # Analytic collision agrees with densely sampled simultaneous motion.
    for _ in range(100):
        a, b, c, d = rng.normal(size=(4, 3))
        t, gap, _ = _swept_contact(a, b, c, d, .16)
        samples = np.linspace(0, 1, 10001)[:, None]
        sampled_gap = np.linalg.norm(a - c + samples * (b - a - d + c), axis=1).min()
        assert abs(gap - sampled_gap) < 1e-5
        if t is not None and t > 0:
            assert np.isclose(np.linalg.norm(a - c + t * (b - a - d + c)), .16)

    # No hovering reward exploit; verify discounted potential telescoping.
    env = HuntEnv(config={"stationary_moth": True})
    env.reset(seed=1)
    env.bat.position[:] = (0, 0, 1.5)
    env.bat.velocity[:] = 0
    env.bat.yaw = env.bat.pitch = 0
    env.moth.position[:] = (2, 0, 1.5)
    sensor = env.observe("bat")
    phi = -.1 * sensor[10] + .3 * max(0, float(sensor[6:9] @ sensor[11:14]))
    shaping, total = 0., 0.
    for tick in range(70):
        _, reward, done, _, info = env.step_with_actions({"bat": [0, 0, -1], "moth": [0, 0, -1]})
        shaping += .99 ** tick * (info["bat_distance_reward"] + info["bat_alignment_reward"])
        total += reward
        if done:
            break
    assert not info["caught"] and total < 0
    assert np.isclose(shaping, -phi)

    # Sonar coordinate transform preserves masking and rotates consistently.
    obs = np.zeros(19, dtype=np.float32)
    obs[6:9], obs[11:14] = (0, 1, 0), (-1, 0, 0)
    assert np.allclose(bat_policy_observation(obs)[11:14], (0, 1, 0))
    obs[11:17] = 0
    assert not np.any(bat_policy_observation(obs)[11:17])

    model = CognitiveActorCritic("bat", 19)
    sequence = torch.randn(1, 30, 19)
    with torch.no_grad():
        full = model.forward_sequence(sequence)
        hidden = model.initial_hidden()
        means = []
        for tick in range(30):
            step = model.forward_sequence(sequence[:, tick:tick+1], hidden)
            means.append(step.mean)
            hidden = step.hidden
        assert torch.allclose(torch.cat(means, 1), full.mean, atol=1e-6)

    replay = EpisodeReplayBuffer(200)
    for _ in range(4):
        states = rng.normal(size=(31, 19)).astype(np.float32)
        replay.push(dict(obs=states[:-1], next_obs=states[1:], actions=rng.uniform(-1, 1, (30, 3)),
            rewards=rng.normal(size=30), dones=np.r_[np.zeros(29), 1],
            next_own=states[1:, :6], opponent=np.zeros((30, 6)),
            visible=np.ones(30, dtype=bool), occluded=np.zeros(30, dtype=bool)))
    sample = replay.sample_segments(1, 8, np.random.default_rng(3))
    assert sample["valid"].sum() == 8
    assert sample["obs"].shape[1] > 8  # prefix retained, not reset at loss window
    assert any(np.array_equal(sample["obs"][0, 0], ep["obs"][0]) for ep in replay.episodes)
    restored = EpisodeReplayBuffer(1)
    restored.load_state_dict(replay.state_dict())
    for key, value in replay.sample_segments(2, 8, np.random.default_rng(9)).items():
        assert np.array_equal(value, restored.sample_segments(2, 8, np.random.default_rng(9))[key])

    critic = TwinCritic()
    target = copy.deepcopy(critic).requires_grad_(False)
    alpha = torch.tensor(-2.3, requires_grad=True)
    old_encoder = critic.encoders[0].weight_ih_l0.detach().clone()
    old_gate = model.workspace_score.weight.detach().clone()
    result = _sac_update(model, critic, target, alpha,
        torch.optim.Adam(model.parameters(), lr=.001),
        torch.optim.Adam(critic.parameters(), lr=.001),
        torch.optim.Adam([alpha], lr=.001), replay, -3, rng)
    assert all(np.isfinite(v) for v in result.values())
    assert not torch.equal(old_encoder, critic.encoders[0].weight_ih_l0)
    assert not torch.equal(old_gate, model.workspace_score.weight)
    assert all(p.grad is None for p in target.parameters())
    print("PASS: SI equivalence, simultaneous contact, reward anti-farming, recurrent history, replay restore, SAC gradients")


if __name__ == "__main__":
    checks()
