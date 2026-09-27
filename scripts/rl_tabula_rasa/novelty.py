"""Dynamic action-sequence repertoire used for Phase 2 novelty measurement."""

from __future__ import annotations

from collections import deque

import numpy as np

from .env import HuntEnv


REPERTOIRE_SIZE = 50
NOVELTY_WEIGHT = 0.05
NOVELTY_THRESHOLD = 0.4
REFERENCE_LENGTH = 12
NOVELTY_WARNING_RATE = 0.95


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-8 else 0.0


def action_signature(actions: list[np.ndarray]) -> np.ndarray:
    """Resample a variable-length episode to a comparable fixed-size signature."""
    if not actions:
        return np.zeros((REFERENCE_LENGTH, 3), dtype=np.float32).ravel()
    sequence = np.asarray(actions, dtype=np.float32).reshape(-1, 3)
    indices = np.linspace(0, len(sequence) - 1, REFERENCE_LENGTH).astype(int)
    return sequence[indices].ravel()


def reference_trajectories(role: str) -> dict[str, np.ndarray]:
    """Retired hand-designed patterns, retained only as a legacy diagnostic."""
    t = np.arange(REFERENCE_LENGTH, dtype=np.float32)
    straight = np.column_stack((np.zeros_like(t), np.zeros_like(t), np.ones_like(t)))
    sweep = np.column_stack((0.7 * np.sin(t * np.pi / 3), np.zeros_like(t), np.ones_like(t)))
    if role == "bat":
        return {"DIRECT": straight,
                "LEAD": np.column_stack((0.2 * np.ones_like(t), np.zeros_like(t), np.ones_like(t))),
                "SWEEP": sweep}
    return {"LIGHT": straight,
            "DODGE": np.column_stack((np.where(t < 5, 0.0, 1.0), np.zeros_like(t), np.ones_like(t))),
            "FLANK": np.column_stack((0.45 * np.ones_like(t), np.zeros_like(t), np.ones_like(t))),
            "LATE": np.column_stack((np.where(t < 9, 0.0, -1.0), np.zeros_like(t), np.ones_like(t))),
            "FEINT": np.column_stack((np.where(t < 6, 0.7, -0.7), np.zeros_like(t), np.ones_like(t)))}


def legacy_reference_similarity(signature: np.ndarray, role: str) -> float:
    return max(cosine_similarity(signature, reference.ravel())
               for reference in reference_trajectories(role).values())


class BehavioralRepertoire:
    """Bounded, per-role archive of distinct complete action-sequence signatures."""

    def __init__(self, capacity: int = REPERTOIRE_SIZE,
                 threshold: float = NOVELTY_THRESHOLD):
        self.capacity = capacity
        self.threshold = threshold
        self.entries: dict[str, list[np.ndarray]] = {"bat": [], "moth": []}

    def nearest_similarity(self, role: str, signature: np.ndarray) -> float | None:
        entries = self.entries[role]
        if not entries:
            return None
        return max(cosine_similarity(signature, entry) for entry in entries)

    def classify_and_add(self, role: str, signature: np.ndarray) -> tuple[bool, float, float | None]:
        """Return (added, cosine-distance, nearest-similarity-before-add)."""
        entries = self.entries[role]
        nearest = self.nearest_similarity(role, signature)
        distance = 1.0 if nearest is None else 1.0 - nearest
        novel = nearest is None or nearest < self.threshold
        if not novel:
            return False, float(distance), nearest

        candidate = np.asarray(signature, dtype=np.float32).copy()
        if len(entries) < self.capacity:
            entries.append(candidate)
        else:
            # Evict the least distinctive member: lowest mean pairwise cosine
            # distance to the rest of the repertoire.
            if len(entries) == 1:
                victim = 0
            else:
                matrix = np.asarray([[1.0 - cosine_similarity(a, b) for b in entries]
                                     for a in entries], dtype=np.float32)
                np.fill_diagonal(matrix, np.nan)
                scores = np.nanmean(matrix, axis=1)
                victim = int(np.nanargmin(scores))
            entries[victim] = candidate
        return True, float(distance), nearest

    def state(self) -> dict:
        return {"capacity": self.capacity, "threshold": self.threshold,
                "entries": {role: [entry.tolist() for entry in entries]
                            for role, entries in self.entries.items()}}

    def load_state(self, state: dict) -> None:
        self.capacity = int(state.get("capacity", self.capacity))
        self.threshold = float(state.get("threshold", self.threshold))
        self.entries = {role: [np.asarray(entry, dtype=np.float32)
                               for entry in state.get("entries", {}).get(role, [])]
                        for role in ("bat", "moth")}


def trajectory_novelty(actions: list[np.ndarray], role: str) -> tuple[bool, float]:
    """Compatibility helper for tests/tools: legacy score, not benchmark novelty."""
    if not actions:
        return False, 1.0
    score = legacy_reference_similarity(action_signature(actions), role)
    return score < NOVELTY_THRESHOLD, score


class NoveltyEnv(HuntEnv):
    def __init__(self, role="bat", opponent=None, config=None):
        super().__init__(role, opponent, config)
        self.action_buffer = {agent: deque(maxlen=REFERENCE_LENGTH) for agent in ("bat", "moth")}
        self.trajectory = {agent: [] for agent in ("bat", "moth")}
        self.repertoire = BehavioralRepertoire()

    def reset(self, *, seed=None, options=None):
        self.action_buffer = {agent: deque(maxlen=REFERENCE_LENGTH) for agent in ("bat", "moth")}
        self.trajectory = {agent: [] for agent in ("bat", "moth")}
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        for role, command in self.last_actions.items():
            self.action_buffer[role].append(np.asarray(command, dtype=np.float32).copy())
            self.trajectory[role].append(np.asarray(command, dtype=np.float32).copy())

        if done:
            episode_metrics = {}
            for role in ("bat", "moth"):
                signature = action_signature(self.trajectory[role])
                added, distance, nearest = self.repertoire.classify_and_add(role, signature)
                episode_metrics[role] = {
                    "novel_behavior": added,
                    "novelty_distance": distance,
                    "nearest_repertoire_similarity": nearest,
                    "legacy_reference_similarity": legacy_reference_similarity(signature, role),
                }
            info["behavior_metrics"] = episode_metrics
            info["novel_behavior"] = {role: episode_metrics[role]["novel_behavior"]
                                      for role in episode_metrics}
            info["legacy_reference_similarity"] = {
                role: episode_metrics[role]["legacy_reference_similarity"]
                for role in episode_metrics}
            # One small terminal bonus per episode keeps its maximum magnitude
            # (0.05 * cosine distance) low relative to catch/survival outcomes.
            info["novelty_bonus"] = NOVELTY_WEIGHT * episode_metrics[self.role]["novelty_distance"]
            reward += info["novelty_bonus"]
        else:
            info["novelty_bonus"] = 0.0
        return obs, reward, done, truncated, info

    def step_joint(self, actions: dict[str, np.ndarray]):
        """Advance one episode with both PPO actions and return per-agent data."""
        _, _, done, truncated, info = HuntEnv.step_with_actions(self, actions)
        for role, command in actions.items():
            self.trajectory[role].append(np.asarray(command, dtype=np.float32).copy())

        bonuses = {"bat": 0.0, "moth": 0.0}
        if done:
            episode_metrics = {}
            for role in ("bat", "moth"):
                signature = action_signature(self.trajectory[role])
                added, distance, nearest = self.repertoire.classify_and_add(role, signature)
                episode_metrics[role] = {
                    "novel_behavior": added,
                    "novelty_distance": distance,
                    "nearest_repertoire_similarity": nearest,
                    "legacy_reference_similarity": legacy_reference_similarity(signature, role),
                }
                bonuses[role] = NOVELTY_WEIGHT * distance
            info["behavior_metrics"] = episode_metrics
            info["novel_behavior"] = {role: data["novel_behavior"]
                                      for role, data in episode_metrics.items()}
            info["legacy_reference_similarity"] = {
                role: data["legacy_reference_similarity"]
                for role, data in episode_metrics.items()}
        info["novelty_bonus_by_role"] = bonuses
        observations = {role: self.observe(role) for role in ("bat", "moth")}
        rewards = {role: float(info[f"{role}_reward"] + bonuses[role])
                   for role in ("bat", "moth")}
        return observations, rewards, done, truncated, info

    def novelty_state(self) -> dict:
        return {"repertoire": self.repertoire.state()}

    def load_novelty_state(self, state: dict):
        # Read the prior rolling-buffer shape only for backwards compatibility;
        # it is not used as the behavioral repertoire or as a benchmark metric.
        if "repertoire" in state:
            self.repertoire.load_state(state["repertoire"])
