# Preserved 37% flat SAC checkpoint

Local safety copy made September 27, 2026. This is not an off-device backup.
Do not train into this directory. Copy the run files to a new output stem before
resuming; retain this evaluated checkpoint unchanged.

- Algorithm: flat Stable-Baselines3 SAC, not the cognitive architecture.
- Seed 23; 60,000 environment steps; 59,000 SAC updates.
- Initial curriculum: 20,000 steps. The full configuration is in the run JSON.
- Fresh test: 74/200 mixed-opponent catches (37%); stationary: 16/200 (8%).
- ZIP contains policy, critics, entropy state, and optimizer state.
- Replay file contains 20,000 transitions and was successfully loaded with the ZIP.
- Frozen moth dependency remains at
  `data/cognitive_swept_batched_seed23_1000ep_20260926.pt` in the repository.
  This bundle is not a standalone environment/source distribution.
- Resume starts a new hunt; it is not bit-identical mid-episode continuation.

Verify this copy with `shasum -a 256 -c SHA256SUMS` from this directory.
The original working files remain in `results/`; no GitHub upload was performed.
