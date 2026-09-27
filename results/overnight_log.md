# Overnight improvement log

## 2026-09-28 — baseline and scope

The README and high-effort report were read before changes. The repository was
clean at commit `2f64212`, with the verified frozen-bat baseline at 39.5%
mixed catch rate and the biological-versus-pure-RL moth comparison at 16.75%
versus 22.25% across 400 held-out hunts.

The required pre-change checks passed:

| Check | Result |
|---|---|
| `test_env` | passed |
| `test_sac` | passed |
| `test_novelty` | passed |
| `cognitive` self-check | passed |

Priority 1 will be implemented first. Validated SAC checkpoints and
`results/benchmark.json` are out of scope and will remain unchanged. No push
to `origin` is authorized for this task.

## 2026-09-28 — league smoke failure and fix

The first three-episode league smoke run reached checkpoint evaluation but
failed when loading a historical moth because `set_training_mode` was called
on the SB3 `SAC` wrapper instead of its policy. The fix changed the call to
`historical_moth.policy.set_training_mode(False)`. The failure was not treated
as a training result.

The corrected smoke run passed with a two-snapshot pool and forced historical
selection after the pool became nonempty. A resume smoke run also passed and
restored the league file before selecting a historical moth. The required
tests and the new league bookkeeping test passed after the fix.

## 2026-09-28 — priority 1 league result

The five-snapshot league experiment completed 1,000 episodes with 200-seed
evaluations on the fixed held-out range `60000–60199` at every 100-episode
checkpoint. It selected the current moth for 822 episodes and historical moths
for 178 episodes. Catch rate was 20.5% at episode 100, 11.5% at episode 500,
and 7.0% at episode 1,000. The 30% threshold was not exceeded at any
checkpoint, so the two-consecutive-checkpoint stop condition was not met.
The result does not support stable co-evolution from this league mixture.

The full checkpoint evaluations are preserved in
`results/league_coevolution_seed23_20260928.json`.

## 2026-09-28 — priority 2 test failure and fix

The first workspace-dominant self-check failed because the helper passed a
weight list as Python's `max` key function. The fix changed the selector to
index the list through a lambda. No training result or live-state artifact was
created by the failed check.
