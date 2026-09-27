"""Create an aggregated optic -> visual-projection seed for the PPO moth."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

from rl_tabula_rasa.connectome import ANNOTATION_URL, aggregate_visual_pathway


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path,
                        default=ROOT / "data/flywire_annotations_v783.tsv")
    parser.add_argument("--connectivity", type=Path,
                        default=ROOT / "data/2025_Connectivity_783.parquet")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/connectome_seed.json")
    args = parser.parse_args()
    if not args.connectivity.is_file():
        parser.error(f"FlyWire v783 connectivity file not found: {args.connectivity}")
    if not args.annotations.exists():
        args.annotations.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading official neuron annotations to {args.annotations}", flush=True)
        urllib.request.urlretrieve(ANNOTATION_URL, args.annotations)

    annotation_bytes = args.annotations.read_bytes()
    annotations = pd.read_csv(args.annotations, sep="\t", usecols=[
        "root_id", "super_class", "cell_type"])
    edges = pd.read_parquet(args.connectivity, columns=[
        "Presynaptic_ID", "Postsynaptic_ID", "Connectivity"])
    matrix, metadata = aggregate_visual_pathway(annotations, edges)
    metadata.update({"annotation_url": ANNOTATION_URL,
                     "annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
                     "annotation_rows": int(len(annotations)),
                     "connectivity_file": args.connectivity.name})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metadata": metadata,
                                       "weights": matrix.tolist()}, indent=2) + "\n")
    print(f"Saved {matrix.shape[0]}x{matrix.shape[1]} connectome seed to {args.output}")
    print(f"Pathway: {metadata['pathway_edges']:,} edge rows / "
          f"{metadata['pathway_synapses']:,} synapses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
