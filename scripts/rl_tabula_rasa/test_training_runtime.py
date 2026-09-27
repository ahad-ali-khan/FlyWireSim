"""Small persistence and pacing checks for the headless trainer runtime."""

import json
import tempfile
from pathlib import Path

from .training_runtime import TickPacer, atomic_json


def test_runtime_helpers():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "state.json"
        atomic_json(path, {"step": 3, "values": [1.0, 2.0]})
        assert json.loads(path.read_text()) == {"step": 3, "values": [1.0, 2.0]}
    assert TickPacer(enabled=False).pace(lambda: False)


if __name__ == "__main__":
    test_runtime_helpers()
    print("Training runtime checks passed")
