# High-effort experiments — 2026-09-28

These are held-out evaluations, not training-set scores. Each evaluation uses
200 seeds beginning at `60000`, with fresh randomized spawns and obstacle
layouts.

## Stationary-target curriculum

Command:

```bash
uv run python -m scripts.rl_tabula_rasa.train_sac_control \
  --steps 80000 --curriculum-steps 60000 \
  --curriculum-profile stationary --seed 23 \
  --moth-opponent-checkpoint \
    data/cognitive_swept_batched_seed23_1000ep_20260926.pt \
  --output results/sac_stationary_curriculum_seed23_20260927.json
```

| Evaluation | Catch rate |
|---|---:|
| Standard learned/random/stationary mixture | 39.5% |
| Stationary moth only | 22.5% |

This improves the original frozen-moth stationary result of 8.0% while keeping
the standard mixed-opponent comparison available.

## Simultaneous SAC co-evolution

`train_coevolution_sac.py` controls one shared `HuntEnv`. Bat and moth choose
actions from the same pre-step observations, the environment advances both,
both replay buffers receive that transition, and both SAC learners update
before the next episode. Every episode emits:

```text
self-play: bat step N, moth step N
```

The first pure current-policy arm (1,000 episodes) scored 7.0% against its
learned moth and 8.0% against a stationary moth. A named 200-episode
stationary-release curriculum was then tested; it improved the best 500-episode
checkpoint to 12.0% against its learned moth and 26.5% against a stationary
moth. The best pair is checked in under
`data/validated_coevolution_stationary_release_20260928/`. Continuing that
same checkpoint to 1,000 episodes regressed to 4.5% and 6.0% respectively.
The result is therefore retained as an honest instability finding, not
presented as a successful co-evolution claim.

## Connectome initialization

The connectome-seeded moth remained a null result: 99.6% survival (498/500)
versus 99.4% (497/500) for random initialization on the same held-out arm.
The one-episode difference is not evidence of a biological initialization
benefit.
