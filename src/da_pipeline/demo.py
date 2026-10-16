"""``da-pipeline-demo``: generate a synthetic dataset and analyse it.

One command that takes a clone of this repository from "installed" to "a
directory full of figures", with no data of your own:

    $ da-pipeline-demo

It writes synthetic extract files, the JSON config describing them, and then
runs the real pipeline over them -- the same code path a real dataset takes.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import PipelineConfig
from .io.paths import create_output_dir, setup_logging
from .synthetic import (
    DEFAULT_CONDITIONS,
    DEFAULT_N_DEGRADATION_METRICS,
    generate_dataset,
)

logger = logging.getLogger(__name__)

DEFAULT_DEMO_DIR = Path("demo_data")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="da-pipeline-demo",
        description="Generate a synthetic dataset and run the pipeline over it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_DEMO_DIR,
        help="Directory for the synthetic extracts, the config and the results.",
    )
    parser.add_argument(
        "--observations_per_group",
        type=int,
        default=120,
        help="Observations per group per file. Raise it for a slower, more realistic run.",
    )
    parser.add_argument(
        "--n_degradation_metrics",
        type=int,
        default=DEFAULT_N_DEGRADATION_METRICS,
        help="How many degradation metrics to plant in each file. The format fixes "
             "neither the count nor the names, so any number is valid; three is "
             "enough for the figures to show structure.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed; the same seed writes the same dataset.",
    )
    parser.add_argument(
        "--generate_only",
        action="store_true",
        help="Write the dataset and config, but do not run the pipeline.",
    )
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)

    root = args.output_dir.resolve()

    print(f"Generating synthetic dataset in {root} ...")
    config_path = generate_dataset(
        root,
        conditions=DEFAULT_CONDITIONS,
        observations_per_group=args.observations_per_group,
        n_degradation_metrics=args.n_degradation_metrics,
        seed=args.seed,
    )

    n_files = sum(condition.n_runs for condition in DEFAULT_CONDITIONS)
    print(f"Wrote {n_files} extract files across {len(DEFAULT_CONDITIONS)} conditions.")
    print(f"Config: {config_path}")

    if args.generate_only:
        print("\n--generate_only was set; skipping the run. To analyse it:")
        print(f"  da-pipeline --data_analytics_config_fpath {config_path} --response_name auc")
        return

    config = PipelineConfig(config_path=config_path, response_name="auc")
    config = config.with_output_dir(create_output_dir(config))

    setup_logging(output_dir=config.resolved_output_dir)

    print(f"\nRunning the pipeline. Results will be in {config.resolved_output_dir}\n")

    # Imported here so that --generate_only never pays for matplotlib.
    from .pipeline import run

    output_dir = run(config)

    print(f"\nDone. Figures and CSVs are in:\n  {output_dir}")


if __name__ == "__main__":
    main()
