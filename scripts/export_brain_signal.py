import json
from pathlib import Path

import numpy as np
import pandas as pd


root = Path(__file__).resolve().parents[1]
trace = root / "data/results/brian2cpp_t0.1s_n1.parquet"
out = root / "data/brain_signal.json"
spikes = pd.read_parquet(trace, columns=["time_ms", "flywire_id"])
left = np.histogram(
    spikes.loc[spikes.flywire_id.eq(720575940614763666), "time_ms"],
    np.linspace(0, 100, 101),
)[0]
right = np.histogram(
    spikes.loc[spikes.flywire_id.isin([720575940631997032, 720575940647030324]), "time_ms"],
    np.linspace(0, 100, 101),
)[0]
signal = (right - left).astype(float)
peak = max(float(np.max(np.abs(signal), initial=0)), 1.0)
out.write_text(json.dumps(np.clip(signal / peak, -1, 1).tolist()))
print(out)
