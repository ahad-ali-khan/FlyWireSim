"""Checks that live workspace weights resolve to named bat/moth leaders."""

from .cognitive import workspace_dominants_by_role


def test_workspace_dominants():
    result = workspace_dominants_by_role({
        "bat": {"sensory": 0.15, "proprioception": 0.55,
                "self_model": 0.20, "belief_state": 0.10},
        "moth": {"sensory": 0.10, "proprioception": 0.10,
                 "self_model": 0.15, "belief_state": 0.65},
    })
    assert result == {"bat": "proprioception", "moth": "belief_state"}


if __name__ == "__main__":
    test_workspace_dominants()
    print("workspace HUD checks passed")
