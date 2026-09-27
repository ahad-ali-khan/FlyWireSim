"""Build and apply a small, auditable FlyWire visual-pathway initialization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


VISUAL_OBSERVATION_INDICES = tuple(range(9, 17))
VISUAL_INPUT_LABELS = (
    "bat_visible", "bat_direction_x", "bat_direction_y", "bat_direction_z",
    "bat_distance", "relative_velocity_x", "relative_velocity_y", "relative_velocity_z",
)
ANNOTATION_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
    "supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)
CONNECTOME_SEEDED = False


def aggregate_visual_pathway(annotations: pd.DataFrame, edges: pd.DataFrame,
                             outputs: int = 64) -> tuple[np.ndarray, dict]:
    """Summarize real optic -> visual-projection synapses by annotated cell type.

    The output rows are projection cell types; columns are optic-lobe cell types.
    The resulting matrix is normalized to a Xavier-like scale before it is applied
    to the moth's eight vision-derived observation channels.
    """
    required_annotations = {"root_id", "super_class", "cell_type"}
    required_edges = {"Presynaptic_ID", "Postsynaptic_ID", "Connectivity"}
    if missing := required_annotations - set(annotations.columns):
        raise ValueError(f"annotation table missing columns: {sorted(missing)}")
    if missing := required_edges - set(edges.columns):
        raise ValueError(f"connectivity table missing columns: {sorted(missing)}")
    labels = annotations.dropna(subset=["cell_type"]).copy()
    labels["root_id"] = labels["root_id"].astype("int64")
    optic = labels[labels["super_class"].eq("optic")]
    projection = labels[labels["super_class"].eq("visual_projection")]
    if optic.empty or projection.empty:
        raise ValueError("No annotated optic-lobe or visual-projection neurons found")

    optic_ids = set(optic.root_id)
    projection_ids = set(projection.root_id)
    pathway = edges[edges.Presynaptic_ID.isin(optic_ids)
                   & edges.Postsynaptic_ID.isin(projection_ids)].copy()
    if pathway.empty:
        raise ValueError("No optic -> visual-projection edges in the supplied connectome")
    types = labels.set_index("root_id")["cell_type"]
    pathway["optic_type"] = pathway.Presynaptic_ID.map(types)
    pathway["projection_type"] = pathway.Postsynaptic_ID.map(types)
    grouped = pathway.groupby(["projection_type", "optic_type"], observed=True)[
        "Connectivity"].sum().unstack(fill_value=0)

    # Rank groups only from the local v783 edge data; deterministic tie breaks
    # make the generated initialization reproducible.
    source_types = grouped.sum(axis=0).sort_values(ascending=False, kind="stable").head(8).index.tolist()
    target_types = grouped.sum(axis=1).sort_values(ascending=False, kind="stable").head(outputs).index.tolist()
    counts = grouped.reindex(index=target_types, columns=source_types, fill_value=0)
    logged = np.log1p(counts.to_numpy(dtype=np.float64))
    if np.linalg.norm(logged) <= 1e-12:
        raise ValueError("Selected optic -> visual-projection matrix has no weights")
    # First Linear layer has 22 input and 64 output units. Match its ordinary
    # Xavier-scale magnitude while retaining the relative strengths from the data.
    target_norm = np.sqrt(outputs * len(source_types)) * np.sqrt(2.0 / (22 + outputs))
    weights = logged / np.linalg.norm(logged) * target_norm
    metadata = {
        "dataset": "FlyWire FAFB v783 local connectivity",
        "annotations": "flyconnectome/flywire_annotations Supplemental_file1",
        "pathway": "optic -> visual_projection",
        "pathway_edges": int(len(pathway)),
        "pathway_synapses": int(pathway.Connectivity.sum()),
        "pathway_source_neurons": int(pathway.Presynaptic_ID.nunique()),
        "pathway_target_neurons": int(pathway.Postsynaptic_ID.nunique()),
        "optic_cell_types": source_types,
        "visual_projection_cell_types": target_types,
        "visual_observation_labels": list(VISUAL_INPUT_LABELS),
        "matrix_orientation": "visual_projection_cell_type x optic_cell_type",
        "normalization": "log1p synapse counts; Frobenius norm scaled to Xavier magnitude",
        "mapping_note": "Cell-type aggregation and mapping the eight visual-derived observation channels to ranked optic cell types are explicit modeling choices; these channels are not individual biological neurons.",
    }
    return weights.astype(np.float32), metadata


def load_seed(path: str | Path) -> tuple[np.ndarray, dict]:
    payload = json.loads(Path(path).read_text())
    weights = np.asarray(payload["weights"], dtype=np.float32)
    if weights.shape != (64, len(VISUAL_OBSERVATION_INDICES)):
        raise ValueError(f"expected 64x8 seed matrix, received {weights.shape}")
    if not np.isfinite(weights).all() or not np.any(weights):
        raise ValueError("connectome seed matrix must be finite and non-zero")
    return weights, payload["metadata"]


def apply_seed(model, weights: np.ndarray) -> None:
    """Replace just vision-channel weights in the actor's first Linear layer."""
    layer = model.policy.mlp_extractor.policy_net[0]
    expected = (layer.out_features, len(VISUAL_OBSERVATION_INDICES))
    if tuple(weights.shape) != expected or layer.in_features != 22:
        raise ValueError(f"PPO actor first layer incompatible with seed: {layer.weight.shape}")
    with torch.no_grad():
        layer.weight[:, list(VISUAL_OBSERVATION_INDICES)] = torch.as_tensor(
            weights, dtype=layer.weight.dtype, device=layer.weight.device)
