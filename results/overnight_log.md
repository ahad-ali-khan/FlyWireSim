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

## 2026-09-28 — priority 3 biological pitch result

The biological moth was evaluated with a 0.30-radian upward pitch target
scaled by Giant Fibre activity. On seeds `60000–60199`, catch rate changed
from 17.5% (35/200) to 22.0% (44/200). On `60200–60399`, it changed from
16.0% (32/200) to 21.0% (42/200). The pure-RL control remained at 22.0%
(44/200) and 22.5% (45/200). Mean commanded pitch bias was 0.00067 and
0.00703 radians/tick. The bat checkpoint was unchanged.

## 2026-09-28 — priority 4 paper draft

The paper draft was written from the checked-in result JSONs and the verified
experiment report. It includes the 39.5% mixed and 22.5% stationary curriculum
results, the league regression, the biological comparison, and the pitch-arm
comparison. Claims without a verified numeric source were omitted or marked as
unverified. The draft passed the requested section-length, heading, banned-word,
table, and figure-marker checks. No protected checkpoint or
`results/benchmark.json` was modified.

## 2026-09-28 — abstract framing revision

The paper abstract was revised to lead with the matched-performance result:
21.5% biological catch rate versus 22.25% pure-RL catch rate across 400
held-out hunts, with no biological-controller training in this environment.
The abstract remained within the 150-word limit and retained the earlier
league and curriculum results for context.
