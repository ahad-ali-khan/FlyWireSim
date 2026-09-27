# Validated co-evolution research checkpoint

This directory contains the best 500-episode stationary-release co-evolution
pair. Bat and moth updated from the same shared transition every episode. The
moth was released from stationary-like actions over the first 200 episodes.

Held-out results on seeds 60000–60199:

- 12.0% catch rate against the learned moth
- 26.5% catch rate against a stationary moth

The 1,000-episode continuation regressed to 4.5% and 6.0%, so this is a
research checkpoint, not a stable headline policy. Verify the paired models,
replay buffers, run JSON, and holdout JSON with `SHA256SUMS`.
