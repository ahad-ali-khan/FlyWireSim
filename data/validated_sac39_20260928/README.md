# Validated SAC 39.5% checkpoint

This directory contains the checked-in stationary-curriculum SAC run that
produced **39.5% catch rate on the 200-seed mixed-opponent holdout** and 22.5%
against stationary moths.

- Seed: 23
- Total transitions: 80,000
- Curriculum transitions: 60,000
- Curriculum profile: `stationary`
- Holdout seeds: 60000–60199
- Physics version: `si_flight_synchronous_contact_v1`

The model ZIP, replay buffer, run manifest, and held-out evaluation are paired
by filename. Verify them with `SHA256SUMS`.
