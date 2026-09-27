"""Small synthetic test for the connectome pathway aggregation."""

import pandas as pd
import numpy as np
from stable_baselines3 import PPO

from .connectome import (VISUAL_OBSERVATION_INDICES, aggregate_visual_pathway,
                         apply_seed)
from .env import HuntEnv


def test_aggregate_visual_pathway():
    annotations = pd.DataFrame([
        {"root_id": i, "super_class": "optic", "cell_type": f"optic_{i}"}
        for i in range(1, 9)
    ] + [
        {"root_id": i + 100, "super_class": "visual_projection", "cell_type": f"projection_{i}"}
        for i in range(1, 4)
    ])
    edges = pd.DataFrame([
        {"Presynaptic_ID": i, "Postsynaptic_ID": 101 + (i % 3), "Connectivity": i}
        for i in range(1, 9)
    ] + [
        # Non-pathway edges must not leak into the seed.
        {"Presynaptic_ID": 1, "Postsynaptic_ID": 8, "Connectivity": 99},
    ])
    weights, metadata = aggregate_visual_pathway(annotations, edges, outputs=3)
    assert weights.shape == (3, 8)
    assert weights.any() and (weights >= 0).all()
    assert metadata["pathway_edges"] == 8
    assert metadata["pathway_synapses"] == 36
    assert len(metadata["optic_cell_types"]) == 8

    model = PPO("MlpPolicy", HuntEnv("moth"), n_steps=16, batch_size=8,
                n_epochs=1, verbose=0)
    first = model.policy.mlp_extractor.policy_net[0]
    before = first.weight.detach().cpu().numpy().copy()
    model_seed = np.full((64, 8), 0.125, dtype=np.float32)
    apply_seed(model, model_seed)
    after = first.weight.detach().cpu().numpy()
    assert np.allclose(after[:, list(VISUAL_OBSERVATION_INDICES)], model_seed)
    unseeded = [i for i in range(first.in_features) if i not in VISUAL_OBSERVATION_INDICES]
    assert np.array_equal(after[:, unseeded], before[:, unseeded])


if __name__ == "__main__":
    test_aggregate_visual_pathway()
    print("Connectome seed checks passed")
