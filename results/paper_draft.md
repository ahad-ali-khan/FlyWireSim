FlyWireSim: Bat–Moth Pursuit with Reinforcement Learning and a Connectome-Grounded Escape Circuit

## Abstract

A connectome-grounded LPLC2-to-Giant Fibre escape circuit, augmented with an activity-scaled pitch component, matched a pure-RL moth at 21.5% versus 22.25% catch rate across 400 held-out hunts, without any training in this environment. FlyWireSim studied bat pursuit of a moth in a 3D environment with randomized starts, terrain, obstacles, flame attraction, sonar, vision, and swept jaw contact. A flat SAC bat reached 39.5% catch rate on 200 held-out mixed-opponent hunts and 22.5% against a stationary moth. A simultaneous league experiment with five historical moth snapshots did not stabilize training: held-out catch rate fell from 20.5% at episode 100 to 7.0% at episode 1,000. The unaugmented biological circuit produced 16.75% catch rate across 400 hunts, compared with 22.25% for a pure-RL moth. These results characterize a reproducible environment and several failure modes rather than establish a general advantage for either controller.

## Introduction

Predator–prey pursuit combines partial observation, continuous control, moving collision geometry, and non-stationary opponents. It also provides a compact setting for comparing learned policies with structured biological mechanisms. FlyWireSim was developed as a bat–moth environment where a bat receives sonar-like observations and a moth receives vision and flame-location observations. The bat learns with Soft Actor-Critic (SAC). The project also includes a recurrent four-specialist cognitive policy and a small escape circuit based on the Drosophila LPLC2-to-Giant Fibre pathway.

The biological arm was motivated by work showing that LPLC2 contributes visual looming information to the Giant Fibre escape pathway [1], and that the Giant Fibre integrates looming information across visual locations [2]. The implementation did not claim to reproduce the full fly brain. It used a connectome-verified local edge and an explicit motor readout.

The experiments addressed three questions. First, could curriculum-trained SAC produce a bat policy above random action and approach a deterministic pursuit reference? Second, could a league of historical moth policies prevent the regression observed in simultaneous co-evolution? Third, did a fixed biological escape circuit change bat catch rate relative to a pure-RL moth under matched held-out layouts?

[FIGURE NEEDED: learning curves for flat SAC, curriculum SAC, and league co-evolution]

## Environment

The environment presents two agents in a bounded 3D arena. The verified curriculum configuration uses an arena spanning 3.5 m by 2.7 m in the horizontal axes, 70 ticks per hunt, and a simulation step of 0.16 s. The bat controls yaw, pitch, and thrust from body-frame sonar observations. The moth controls the same three action dimensions from visual and flame-related observations. Natural-environment runs include seeded terrain and obstacles with obstacle density 0.08 obstacles/m². The flame is a persistent ecological target for the moth.

Bat–moth contact is computed from the swept jaw segment and moth segment during each tick. The verified jaw-contact radius is 0.16 m. A catch records the contact fraction and contact position. This geometry replaced endpoint-only checks after earlier diagnostics found missed contacts. The bat receives terminal catch and timeout signals with distance and heading shaping. The moth receives survival and caught signals with distance and flame shaping.

The environment randomizes agent starts subject to a minimum separation and uses held-out seeds to generate new starts and obstacle layouts. Evaluation uses deterministic bat actions and fixed evaluation seeds. Training and evaluation configurations record the physics version `si_flight_synchronous_contact_v1`.

## Methods

The main bat learner uses SAC with continuous three-dimensional actions. The curriculum arm used 80,000 environment steps, with the stationary-target curriculum active for 60,000 steps. The league arm initialized its bat from the validated curriculum policy and initialized a moth learner for simultaneous updates. It retained five historical moth policy snapshots. Each training episode selected the current moth with probability 0.80 or one historical snapshot with probability 0.20. Evaluations used 200 seeds beginning at 60000 every 100 episodes. Training stopped only after two consecutive evaluations exceeded 30% catch rate; this criterion was not reached.

The cognitive arm uses sensory, proprioception, self-model, and belief-state specialists combined by a learned workspace gate. Its recurrent state and workspace weights are exposed through the live-state JSON and Blender HUD. This arm was retained as an architectural comparison and was not used for the headline SAC result.

The biological moth uses a bilateral LPLC2-to-DNp01/Giant Fibre microcircuit. The checked-in connectivity table verified three synapses between the selected roots. Angular-size expansion was approximated from visual radius, closing speed, and squared distance. A normalized leaky integrate-and-fire circuit generated bilateral spike trains. Asymmetric Giant Fibre activity drove turn-away yaw, total activity drove thrust, and a named upward pitch target was tested as a separate arm. These motor gains were engineering choices rather than measured biological weights.

The biological comparison froze the validated bat checkpoint and changed only the moth controller. The first and confirmation evaluations each used 200 seeds, with ranges 60000–60199 and 60200–60399. The same pure-RL moth checkpoint served as the control in both runs.

[FIGURE NEEDED: diagram of the LPLC2-to-DNp01/Giant Fibre readout and matched evaluation protocol]

## Results

The principal bat results were:

| Bat arm | Mixed catch rate | Stationary catch rate |
|---|---:|---:|
| Random action | 5.5% | 0.5% |
| Standard SAC, three seeds | 25.83% ± 8.50 percentage points | not reported in the aggregate JSON |
| Stationary curriculum SAC | 39.5% | 22.5% |
| Deterministic pursuit reference | 41.5% | 14.5% |

The league arm did not satisfy its stability gate. Held-out catch rate was 20.5% at episode 100, 11.5% at episode 500, and 7.0% at episode 1,000. The five-snapshot pool was used, but the decline persisted.

The frozen-bat biological comparison produced 17.5% versus 22.0% catch rate on the first 200-seed range and 16.0% versus 22.5% on the confirmation range for biological and pure-RL moths, respectively. The pitch arm produced 22.0% versus 22.0% and 21.0% versus 22.5%. Across 400 hunts, the pitch arm reached 21.5% and the pure-RL control reached 22.25%.

[FIGURE NEEDED: held-out catch-rate bars with confidence intervals for baseline, league, biological, and pitch arms]

## Limitations

- The league experiment did not reach its 30% stability gate and did not establish stable co-evolution.
- The biological circuit was a small normalized microcircuit, not a full FlyWire brain simulation.
- The pure-RL moth was trained against a specific bat, so the frozen-bat comparison may include opponent-pattern mismatch.
- The biological and pitch comparisons used 200-seed ranges per run; the results were first comparisons rather than definitive superiority tests.
- The cognitive policy was evaluated as an architectural arm and was not the source of the headline SAC result.

## References

1. Ache, J. M., et al. “State-dependent decoupling of sensory and motor circuits underlies behavioral flexibility in Drosophila.” Nature Neuroscience, 2020. PMCID: PMC7444277.
2. Ache, J. M., et al. “Azimuthal invariance to looming stimuli in the Drosophila giant fiber escape circuit.” Journal of Neuroscience, 2023. PMCID: PMC10263144.
3. FlyWire Consortium. “FlyWire whole-brain connectomics resource.” Nature Methods, 2021. The local experiment used the checked-in FAFB v783 connectivity table.
