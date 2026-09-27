"""Fast unit checks for the co-evolution league bookkeeping."""

import numpy as np
import torch
from torch import nn

from .train_coevolution_sac import (append_league_snapshot,
                                    choose_league_opponent, snapshot_policy)


class _Model:
    def __init__(self):
        self.policy = nn.Linear(2, 2)


def main():
    model = _Model()
    first = snapshot_policy(model)
    assert all(value.device.type == "cpu" for value in first.values())
    pool = []
    for episode in range(1, 8):
        with torch.no_grad():
            model.policy.weight.fill_(episode)
        pool = append_league_snapshot(pool, model, episode, max_size=5)
    assert [item["episode"] for item in pool] == [3, 4, 5, 6, 7]
    assert torch.all(pool[-1]["state_dict"]["weight"] == 7)

    mode, snapshot = choose_league_opponent(pool, np.random.default_rng(4), 0.0)
    assert mode == "current" and snapshot is None
    mode, snapshot = choose_league_opponent(pool, np.random.default_rng(4), 1.0)
    assert mode == "historical" and snapshot["episode"] in {3, 4, 5, 6, 7}
    print("league bookkeeping checks passed")


if __name__ == "__main__":
    main()
