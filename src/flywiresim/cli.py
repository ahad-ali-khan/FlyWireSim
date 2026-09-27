from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the real FlyWire brain model from this repository."
    )
    parser.add_argument("--t-run", type=float, default=0.1, help="seconds")
    parser.add_argument("--experiment", default="sugar")
    args, rest = parser.parse_known_args()

    sys.path.insert(0, str(ROOT / "code"))
    from benchmark import BenchmarkLogger, get_experiment, run_benchmarks

    logger = BenchmarkLogger(log_file=None)
    try:
        run_benchmarks(
            backends=["cpu"],
            t_run_values=[args.t_run],
            n_run_values=[1],
            experiment=get_experiment(args.experiment),
            logger=logger,
            rounds=1,
            run_label=None,
            round_start=1,
        )
    finally:
        logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
