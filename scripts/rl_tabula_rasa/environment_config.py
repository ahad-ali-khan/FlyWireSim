"""Shared Phase 5 and Phase 8 scene controls; stdlib-only for Blender imports."""

import math
import random

TERRAIN_DISPLACEMENT_SCALE = 0.22
TERRAIN_NOISE_FREQUENCY = 1.1
OBSTACLE_DENSITY = 0.08
OBSTACLE_JITTER = 0.35
OBSTACLE_SEED = None

TREE_TRUNK_RADIUS = 0.12
TREE_TRUNK_HEIGHT = 2.0
TREE_BRANCH_COUNT = 2
TREE_BRANCH_LENGTH = 0.8
TREE_BRANCH_RADIUS = 0.04

FOG_START_DISTANCE = 6.0
FOG_END_DISTANCE = 18.0
FOG_DENSITY = 0.025
FOG_ANISOTROPY = 0.3
ATMOSPHERE_ENABLED = False

JAW_CONTACT_RADIUS = 0.16
SLOW_MOTION_NEAR_MISS_MULTIPLIER = 2.5
SLOW_MOTION_NEAR_MISS_DISTANCE = JAW_CONTACT_RADIUS * SLOW_MOTION_NEAR_MISS_MULTIPLIER
SLOW_MOTION_DURATION = 12
SLOW_MOTION_SPEED = 0.25
SLOWMO_FOG_DENSITY_MULTIPLIER = 0.0


def sample_obstacle_sites(seed, arena_x=3.5, arena_y=2.7,
                          density=OBSTACLE_DENSITY, jitter=OBSTACLE_JITTER):
    """Shared deterministic tree locations for the physics and Blender view."""
    area = 4 * arena_x * arena_y
    count = int(round(area * density))
    nx = max(1, int(round(math.sqrt(count * arena_x / arena_y))))
    ny = max(1, math.ceil(count / nx))
    xs = [-arena_x + (2 * i + 1) * arena_x / nx for i in range(nx)]
    ys = [-arena_y + (2 * i + 1) * arena_y / ny for i in range(ny)]
    sites = [(x, y) for y in ys for x in xs]
    rng = random.Random(seed)
    rng.shuffle(sites)
    return [(max(-arena_x + 0.3, min(arena_x - 0.3, x + rng.uniform(-jitter, jitter))),
             max(-arena_y + 0.3, min(arena_y - 0.3, y + rng.uniform(-jitter, jitter))),
             rng.uniform(-math.pi, math.pi)) for x, y in sites[:count]]
