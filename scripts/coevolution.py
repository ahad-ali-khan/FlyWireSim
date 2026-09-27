"""Small online Q learners and sensor neurons for the Blender predator-prey demo.

The actions are hand-designed motor primitives. Learning selects and composes
them; it does not train the fixed FlyWire connectome or invent new anatomy.
"""

import random
from collections import deque


class AdaptivePolicy:
    def __init__(self, primitives, saved=None, rng=None):
        saved = saved if (saved or {}).get("version") == 2 else {}
        self.primitives = tuple(primitives)
        self.actions = list(self.primitives) + [name for name in saved.get("hybrids", []) if "+" in name]
        self.weights = {name: list(saved.get("weights", {}).get(name, [0.0] * 5)) for name in self.actions}
        self.counts = {name: int(saved.get("counts", {}).get(name, 0)) for name in self.actions}
        self.episodes = int(saved.get("episodes", 0))
        self.decisions = int(saved.get("decisions", 0))
        self.rng = rng or random.Random()
        self.current = None
        self.previous_features = None
        self.accumulated_reward = 0.0
        self.age = 0
        self.failure_streak = 0
        self.avoid_next = None
        self.episode_usage = {name: 0 for name in self.actions}

    def value(self, name, features):
        return sum(weight * feature for weight, feature in zip(self.weights[name], features))

    def choose(self, features):
        choices = [name for name in self.actions if name != self.avoid_next] or self.actions
        epsilon = max(0.10, 0.32 / (1 + self.decisions / 250))
        if self.rng.random() < epsilon:
            selected = self.rng.choice(choices)
        else:
            selected = max(choices, key=lambda name: self.value(name, features) + self.rng.random() * 0.01)
        changed = selected != self.current
        self.current = selected
        self.previous_features = tuple(features)
        self.accumulated_reward = 0.0
        self.age = 0
        self.avoid_next = None
        self.counts[selected] += 1
        self.episode_usage[selected] = self.episode_usage.get(selected, 0) + 1
        self.decisions += 1
        return changed

    def step(self, features, reward=0.0, terminal=False):
        if self.current is None:
            return self.choose(features)
        self.accumulated_reward += reward
        self.age += 1
        if self.age < 12 and not terminal:
            return False
        old = self.value(self.current, self.previous_features)
        future = 0.0 if terminal else 0.92 * max(self.value(name, features) for name in self.actions)
        error = max(-2.0, min(2.0, self.accumulated_reward + future - old))
        self.weights[self.current] = [
            max(-3.0, min(3.0, weight + 0.11 * error * feature))
            for weight, feature in zip(self.weights[self.current], self.previous_features)
        ]
        self.failure_streak = self.failure_streak + 1 if self.accumulated_reward < -0.005 else 0
        if terminal:
            used = sum(self.episode_usage.values()) or 1
            for name, count in self.episode_usage.items():
                if count:
                    self.weights[name][0] = max(-3.0, min(3.0, self.weights[name][0] + 0.25 * reward * count / used))
            self.episodes += 1
            self.avoid_next = max(self.episode_usage, key=self.episode_usage.get) if reward < 0 else None
            self.current = None
            self.episode_usage = {name: 0 for name in self.actions}
            self._discover()
            return True
        if self.failure_streak >= 2:
            self.avoid_next = self.current
            self.failure_streak = 0
        return self.choose(features)

    def _discover(self):
        # Compose the two best-tested primitives; the new action must earn its place.
        if self.episodes < 8 or len(self.actions) >= len(self.primitives) + 2:
            return
        tested = [name for name in self.primitives if self.counts[name] >= 2]
        if len(tested) < 2:
            return
        ranked = sorted(tested, key=lambda name: self.value(name, (1, 0.5, 0.5, 0.5, 0.5)), reverse=True)
        name = "+".join(sorted(ranked[:2]))
        if name in self.actions:
            return
        self.actions.append(name)
        self.weights[name] = [(a + b) / 2 for a, b in zip(self.weights[ranked[0]], self.weights[ranked[1]])]
        self.counts[name] = 0
        self.episode_usage[name] = 0

    def state(self):
        return {
            "version": 2, "weights": self.weights, "counts": self.counts,
            "hybrids": [name for name in self.actions if "+" in name],
            "episodes": self.episodes, "decisions": self.decisions,
        }


class SpikeSensors:
    """Leaky integrate-and-fire *simulated sensors*, not FlyWire neurons."""

    def __init__(self, names):
        self.voltage = {name: 0.0 for name in names}
        self.history = {name: deque([0] * 96, maxlen=96) for name in names}

    def step(self, inputs):
        for name, stimulus in inputs.items():
            voltage = self.voltage[name] * 0.78 + 0.12 + max(0.0, min(1.0, stimulus)) * 0.68
            spike = int(voltage >= 1.0)
            self.voltage[name] = 0.0 if spike else voltage
            self.history[name].append(spike)


if __name__ == "__main__":
    policy = AdaptivePolicy(["A", "B"], rng=random.Random(3))
    features = (1.0, 0.8, 0.2, 0.0, 0.1)
    for _ in range(24):
        policy.step(features, -0.02)
    assert policy.decisions >= 2 and any(any(w) for w in policy.weights.values())
    most_used = max(policy.episode_usage, key=policy.episode_usage.get)
    policy.step(features, -1.0, terminal=True)
    policy.step(features)
    assert policy.current != most_used and policy.episodes == 1
    sensors = SpikeSensors(["echo"])
    for _ in range(10):
        sensors.step({"echo": 1.0})
    assert sum(sensors.history["echo"]) > 0
    print("coevolution self-check passed")
