"""Small Phase 2 regression check."""

import numpy as np

from .novelty import (BehavioralRepertoire, NoveltyEnv, action_signature,
                      trajectory_novelty)


def test_novelty():
    first = [np.array([1, 0, 0], dtype=np.float32)] * 12
    same = [np.array([1, 0, 0], dtype=np.float32)] * 12
    different = [np.array([0, 1, 0], dtype=np.float32)] * 12
    repertoire = BehavioralRepertoire(capacity=2, threshold=0.4)
    assert repertoire.classify_and_add("bat", action_signature(first))[0]
    assert not repertoire.classify_and_add("bat", action_signature(same))[0]
    assert repertoire.classify_and_add("bat", action_signature(different))[0]
    assert len(repertoire.entries["bat"]) == 2
    third = [np.array([0, 0, 1], dtype=np.float32)] * 12
    assert repertoire.classify_and_add("bat", action_signature(third))[0]
    assert len(repertoire.entries["bat"]) == 2
    saved = repertoire.state()
    restored = BehavioralRepertoire()
    restored.load_state(saved)
    assert len(restored.entries["bat"]) == 2

    env = NoveltyEnv("bat", config={"max_ticks": 1})
    env.reset(seed=7)
    _, reward, done, _, info = env.step(np.array([0, 0, 1], dtype=np.float32))
    assert done
    assert info["novelty_bonus"] > 0
    assert np.isclose(reward, info["bat_reward"] + info["novelty_bonus"])
    assert "legacy_reference_similarity" in info
    assert set(info["novel_behavior"]) == {"bat", "moth"}
    state = env.novelty_state()
    assert state["repertoire"]["entries"]["bat"]
    env.load_novelty_state(state)
    assert env.novelty_state() == state
    assert isinstance(trajectory_novelty(first, "bat")[1], float)


if __name__ == "__main__":
    test_novelty()
    print("Phase 2 novelty checks passed")
