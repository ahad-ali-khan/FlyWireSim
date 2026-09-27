"""Rebuild the repair comparison from saved results; no training or simulation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .training_runtime import atomic_json

RESULTS = Path(__file__).resolve().parents[2] / "results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def main():
    controls = read("repaired_final_matched_holdout_20260927.json")
    rows = []
    for seed in (23, 42, 7):
        result = read(f"sac_standard_control_seed{seed}_holdout_20260927.json")
        rows.append(dict(label=f"Flat SAC · seed {seed}",
            mixed=result["results"][0]["catch_rate"], stationary=result["results"][1]["catch_rate"]))
    rates = [row["mixed"] for row in rows]
    for label, filename in (
        ("Flat SAC + curriculum", "sac_curriculum_control_seed23_holdout_20260927.json"),
        ("Cognitive · episode 400", "repaired_best400_fresh_holdout_20260927.json"),
        ("Cognitive · episode 600", "repaired_final_matched_holdout_20260927.json"),
    ):
        result = read(filename)["results"]
        rows.append(dict(label=label, mixed=result[0]["catch_rate"], stationary=result[1]["catch_rate"]))
    random_rate = next(r["catch_rate"] for r in controls["results"] if r["policy"] == "random" and r["mode"] == "mix")
    pursuit_rate = next(r["catch_rate"] for r in controls["results"] if r["policy"] == "deterministic_pursuit_baseline" and r["mode"] == "mix")
    summary = dict(test_seed_start=60000, test_seeds=200, test_layouts="held out",
        flat_sac_multiseed_mean=float(np.mean(rates)),
        flat_sac_multiseed_sample_std=float(np.std(rates, ddof=1)), rows=rows,
        random_mixed_catch_rate=random_rate, pursuit_mixed_catch_rate=pursuit_rate,
        caveats=["SAC trains only the bat; moth is a frozen 50/30/20 opponent mix.",
                 "Stationary-target pursuit remains weak.",
                 "The cognitive run regresses by episode 600; do not deploy that checkpoint.",
                 "Curriculum uses a longer training budget; not an isolated curriculum ablation.",
                 "FlyWire recorded spikes are separate from these policies."])
    atomic_json(RESULTS / "repair_audit_20260927.json", summary)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), layout="constrained")
    for seed in (23, 42, 7):
        episodes = read(f"sac_standard_control_seed{seed}_20260927.json")["episodes"]
        curve = [100 * np.mean([e["caught"] for e in episodes[max(0, i-99):i+1]]) for i in range(len(episodes))]
        axes[0].plot(np.arange(len(curve))+1, curve, label=f"Flat SAC seed {seed}", lw=1.4)
    episodes = read("cognitive_sac_bodyframe_seed23_600ep.json")["episodes"]
    curve = [100 * np.mean([e["caught"] for e in episodes[max(0, i-99):i+1]]) for i in range(len(episodes))]
    axes[0].plot(np.arange(len(curve))+1, curve, label="Cognitive SAC seed 23", color="#995454", ls="--")
    axes[0].set(title="Training catches · rolling 100 hunts", xlabel="Completed training episodes", ylabel="Catch rate (%)", ylim=(0, 50))
    axes[0].legend(loc="upper left", frameon=False)
    positions = np.arange(len(rows))
    axes[1].barh(positions-.17, [100*r["mixed"] for r in rows], height=.32, color="#267e9c", label="Mixed moth opponents")
    axes[1].barh(positions+.17, [100*r["stationary"] for r in rows], height=.32, color="#cf9745", label="Stationary moth only")
    axes[1].axvline(100*random_rate, color="gray", ls=":", lw=1, label="Random, mixed")
    axes[1].axvline(100*pursuit_rate, color="#667344", ls="--", lw=1, label="Pursuit, mixed")
    axes[1].set(yticks=positions, yticklabels=[r["label"] for r in rows], xlabel="Catch rate (%)",
                title="Fresh test · 200 seeds per condition", xlim=(0, 60))
    axes[1].invert_yaxis()
    for i, row in enumerate(rows):
        axes[1].text(100*row["mixed"]+.6, i-.17, f"{100*row['mixed']:.1f}%", va="center", fontsize=8)
    axes[1].legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("FlyWireSim repair audit — learning gains and remaining failures", fontsize=13)
    fig.supxlabel(
        "Budget note: flat SAC = 30k environment steps; curriculum = 60k (first 20k easier), shown only at right.\n"
        "Both use 1 SAC update per environment step after 1k warm-up steps: 29k vs 59k updates.\n"
        "Cognitive SAC uses 16 recurrent-segment updates per eligible episode (9,296 total). These are not compute-matched runs.",
        fontsize=8,
    )
    fig.savefig(RESULTS / "repair_audit_20260927.png", dpi=160)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
