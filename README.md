# FlyWireSim

A 3D multi-agent pursuit environment where a bat learns to catch a moth using
continuous-control reinforcement learning, biologically-inspired neural
architecture, and physically consistent sonar sensing.

**Best verified result:** 37% catch rate on 200 held-out hunts — 6.7× above a
random baseline — using SAC with curriculum learning.

![Learning curve and held-out evaluation](results/repair_audit_20260927.png)

![Blender wide render](results/blender_scene_render_wide.png)

![Blender close render](results/blender_scene_render_hero.png)

[▶ Watch the 65-second buffered Blender demo](results/flywiresim_buffered_demo_65s.gif)

## What makes this interesting

- **Custom 3D physics environment** — continuous flight, terrain, obstacles,
  boundary forces, moth flame-attraction, and randomized spawns.
- **Swept jaw collision detection** — jaw and moth paths are checked as moving
  bodies each tick; catches record the exact contact fraction and impact point.
  This found and fixed a bug where endpoint-only detection missed 35% of real
  contacts.
- **Biologically-inspired cognitive architecture** — sensory,
  proprioception, self-model, and belief specialists compete for a shared
  64-value workspace through a learned softmax gate. The recurrent state tracks
  occluded opponents. This architecture is preserved as an experimental arm.
- **Rigorous evaluation** — every headline result uses 200 untouched held-out
  seeds with new spawns and obstacle layouts.
- **Blender demo** — 30 consecutive unselected hunts with textured rigged
  models, sonar visualization, near-miss slow motion, and telemetry HUD.

## Results

The bat trains against a frozen moth mixture: 50% learned, 30% random, and
20% stationary. The table below uses 200 held-out seeds.

| Policy | Mixed opponents | Stationary moth |
|---|---:|---:|
| Random actions | 5.5% | 0.5% |
| Flat SAC · seed 23 · 30k steps | 17.5% | 3.5% |
| Flat SAC · seed 42 · 30k steps | 34.5% | 5.5% |
| Flat SAC · seed 7 · 30k steps | 25.5% | 6.0% |
| **Flat SAC + curriculum · 60k steps** | **37.0%** | **8.0%** |
| Deterministic pursuit reference | 41.5% | 14.5% |
| Cognitive SAC · episode 400 | 16.0% | 0.5% |

Three standard SAC seeds average **25.8% ± 8.5 percentage points**. The
curriculum result uses twice the training budget; its advantage reflects both
curriculum and budget. Deterministic pursuit is an omniscient reference
controller, not a theoretical upper bound.

The cognitive architecture peaked at 16% then regressed to 2%; it is preserved
and documented but not used for the demo. Connectome seeding also produced no
measurable advantage in the available comparison: 99.6% versus 99.4% moth
survival on the same 500-hunt held-out arm, a one-episode difference.

Full audit, configurations, failed hypotheses, and limitations:
[`results/repair_audit_20260927.md`](results/repair_audit_20260927.md)

## Quick start

```bash
uv sync
uv run python -m scripts.rl_tabula_rasa.test_env
uv run python -m scripts.rl_tabula_rasa.test_sac
```

Run the buffered Blender playback:

```bash
uv run python scripts/viewer.py --mode live \
  --live-stream-file data/validated_sac_demo_20260927.jsonl \
  --live-state-file data/validated_sac_demo_20260927.state.json \
  --learning-curve data/validated_sac_demo_20260927.curve.json \
  --speed 0.5 --camera chase
```

This playback contains 30 consecutive hunts: 11 catches and 19 timeouts. It
does not update the networks. The HUD identifies frozen evaluation and shows
episode state, jaw-contact outcomes, near-misses, occlusion, sonar/vision
indicators, and the learning curve. The GIF is a low-resolution 65-second
render of those same buffered frames for repository preview.

## Reproduce training

```bash
uv run python -m scripts.rl_tabula_rasa.train_sac_control \
  --steps 60000 --curriculum-steps 20000 --seed 23 \
  --moth-opponent-checkpoint \
    data/cognitive_swept_batched_seed23_1000ep_20260926.pt \
  --output results/my_run.json
```

Resume with the same output stem and a larger `--steps` total. The verified
37% checkpoint is in
[`data/validated_sac37_20260927/`](data/validated_sac37_20260927/) with policy,
critics, optimizer state, replay buffer, and SHA-256 manifest.

## What is implemented

- Continuous bat control with body-frame sonar observations and SAC.
- Randomized spawn positions, seeded natural-environment obstacles, terrain,
  boundary forces, and moth attraction to a warm flame.
- Synchronous swept jaw collision with recorded contact fraction and impact
  point; jaw and moth must be within 0.16 m at the same time.
- Animated, textured bat and moth Blender assets with actual animation clips.
- Bat sonar, moth vision, flame attraction, readable decision HUD, contact dot,
  near-miss slow motion, and buffered JSONL playback.
- Recurrent four-specialist cognitive architecture with sensory,
  proprioception, self-model, and belief heads.
- FlyWire-derived connectome seeding for moth initialization, evaluated as a
  null result rather than presented as a biological advantage.

## Engineering fixes

- Replaced independent path-segment intersection with synchronous relative
  motion, eliminating false catches from paths crossing at different times.
- Applied the `dt=0.16` unit conversion consistently to bat, moth, and boundary
  dynamics; equivalence tests preserve the intended flight envelope.
- Replaced reward farming from repeated heading/thrust bonuses with discounted
  potential shaping whose terminal value is zero.
- Repaired recurrent SAC replay so critic and actor unrolls preserve episode
  history and resume restores replay, optimizers, temperature, and RNG state.

## Honest limitations and next work

- Bat-only learning against a frozen moth; no verified simultaneous co-evolution
  result yet.
- Stationary-target approach remains weak at 8% catch rate.
- Recorded FlyWire/Brian2 spikes are a prerecorded signal, not live policy
  activity.
- Rendering uses visual collision proxies, not exact animated-mesh physics.

The next high-effort phase is to improve stationary capture, then train both
agents with a properly audited co-evolution protocol. The connectome null result
will remain in the report rather than being omitted.

## FlyWire brain tooling

```bash
uv run flywiresim --t-run 0.1 --experiment sugar
uv run python scripts/export_brain_signal.py
/Applications/Blender.app/Contents/MacOS/Blender \
  --factory-startup --python scripts/blender_world.py
```

## Credits and licenses

Built on [FlyWire/fly-brain](https://github.com/eonsystemspbc/fly-brain),
licensed GPL-2.0 like the upstream project.

- Vampire bat model: rubberduck, CC0 —
  [OpenGameArt](https://opengameart.org/content/vampire-bat-animated)
- Moth model: Amatsukast, CC BY-NC-SA 4.0 —
  [Sketchfab](https://sketchfab.com/3d-models/moth-fafcd79b60964e57a532e74706af6d16)

See [`assets/models/creatures/CREDITS.md`](assets/models/creatures/CREDITS.md)
for full attribution. The moth's non-commercial/share-alike terms apply
separately from the upstream code license.
