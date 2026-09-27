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

## Biological looming escape arm

The existing prerecorded SEZ/Brian2 signal was not relabeled as a Giant Fibre
signal. Instead, this arm uses a small causal circuit with the verified local
v783 edge:

```text
angular-size expansion (r * closing_speed / distance²)
  → bilateral LPLC2 drive
  → 5 ms normalized LIF substeps
  → DNp01/Giant Fibre spike rates
  → moth yaw/pitch/thrust action
```

The checked-in connectivity table contains the selected LPLC2 root
`720575940640302389` → DNp01/Giant Fibre root `720575940622838154` edge with
three synapses. The circuit records the full 32-substep bilateral spike train
for every 160 ms environment tick. Its action readout is explicit: left/right
GF rate asymmetry drives turn-away yaw, mean GF rate drives thrust, and a small
upward pitch bias is an engineering choice. When GF activity is quiet, the
existing flame attraction remains as the baseline drive.

### Frozen-bat matched evaluation

The bat checkpoint was frozen at
`data/validated_sac39_20260928/sac_stationary_curriculum_seed23_20260927.zip`.
Both arms use the same bat, 200 held-out seeds (`60000–60199`), natural
environment, randomized obstacle layouts, collision geometry, and deterministic
bat actions. Only the moth controller changes.

| Moth controller | Catch rate | Mean hunt length | Biological telemetry |
|---|---:|---:|---|
| Biological LPLC2 → DNp01/GF | 17.5% (35/200) | 65.14 ticks | 47.49 GF spikes/hunt; mean looming 0.112 |
| Pure-RL moth checkpoint | 22.0% (44/200) | 64.96 ticks | not applicable |

Precisely, the frozen bat caught the biological moth 4.5 percentage points less
often than the pure-RL moth on the first matched holdout. A second independent
200-seed holdout gave 16.0% (32/200) biological versus 22.5% (45/200) pure-RL.
Across both runs the rates are 67/400 (16.75%) and 89/400 (22.25%). The first
run's Wilson 95% intervals are 12.9–23.4% and 16.8–28.2%, so the gap is
interesting but near the uncertainty scale for a 200-seed comparison; it is
not reported as a statistically definitive result.

The RL moth was trained against a specific bat. The biological moth never
trained in this environment and is a fixed controller. Therefore some of the
gap may be bat overfitting to the RL moth's particular patterns and unfamiliarity
with the biological controller, rather than superior biological evasion. The
biological circuit is a connectome-grounded microcircuit abstraction, not a
whole-brain FlyWire simulation. Full per-hunt telemetry is preserved in
`results/biological_escape_vs_pure_rl_20260928.json` and the confirmation run
in `results/biological_escape_vs_pure_rl_confirmatory_20260928.json`.
