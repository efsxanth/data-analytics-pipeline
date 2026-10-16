"""Shared fixtures.

The dataset fixtures write real HDF5 files rather than mocking the reader. The
reader is thin and the schema is the thing most likely to drift, so a test that
mocks it would pass in exactly the situation where the pipeline breaks.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from da_pipeline.config import PipelineConfig, StepRange
from da_pipeline.constants import columns as col
from da_pipeline.metrics import resolve_metric_roles
from da_pipeline.synthetic import (
    ConditionSpec,
    build_group_panel,
    generate_dataset,
    metric_names,
    write_extract,
)

# Small enough that the whole suite runs in seconds, big enough that
# equal-count binning has something to divide.
N_GROUPS = 3
N_STEPS = 20
OBSERVATIONS_PER_GROUP = 40

# The outcome metric the reference producer writes. Tests name it here rather
# than reaching for a constant: which metric is the outcome is configuration,
# so there is deliberately no symbol in constants.columns for it.
OUTCOME_METRIC = "Accuracy"

TEST_CONDITIONS = (
    ConditionSpec("Condition_A", n_runs=2, quality=0.90),
    ConditionSpec("Condition_B", n_runs=2, quality=0.55),
)


@pytest.fixture(scope="session")
def group_panel():
    """The group panel every extract in the session shares."""
    return build_group_panel(n_groups=N_GROUPS, n_steps=N_STEPS, seed=1)


@pytest.fixture(scope="session")
def dataset_root(tmp_path_factory):
    """A complete synthetic dataset on disk; returns the config file path."""
    root = tmp_path_factory.mktemp("dataset")
    return generate_dataset(
        root,
        conditions=TEST_CONDITIONS,
        n_groups=N_GROUPS,
        n_steps=N_STEPS,
        observations_per_group=OBSERVATIONS_PER_GROUP,
        seed=1,
    )


@pytest.fixture(scope="session")
def extract_path(tmp_path_factory, group_panel):
    """A single extract file, for reader-level tests."""
    path = tmp_path_factory.mktemp("single") / "run.h5"
    return write_extract(
        path,
        panel=group_panel,
        run_id="Condition_A.0",
        observations_per_group=OBSERVATIONS_PER_GROUP,
        seed=7,
    )


@pytest.fixture
def config(dataset_root):
    """A config pointing at the synthetic dataset, with defaults elsewhere."""
    return PipelineConfig(
        config_path=dataset_root,
        response_name="auc",
        step_range=StepRange(1, N_STEPS),
        max_workers=2,
    )


# ---------------------------------------------------------------------------
# Hand-built frames, for the analysis modules
# ---------------------------------------------------------------------------

@pytest.fixture
def per_observation_frame():
    """A per-observation accuracy frame with a known, exactly-divisible shape.

    100 observations per condition and 10 bins means equal-count binning must
    produce bins of exactly 10 -- so a test can assert the strict case, not just
    the +/- 1 tolerance.
    """
    rng = np.random.default_rng(0)
    rows = []

    for condition, centre in (("Condition_A", 0.92), ("Condition_B", 0.70)):
        for observation in range(100):
            rows.append({
                col.RUN_ID: f"{condition}.0",
                col.CONDITION: condition,
                col.GROUP_ID: 1 + observation % N_GROUPS,
                col.GROUP_LABEL: f"group_{1 + observation % N_GROUPS}",
                col.OBSERVATION_ID: observation,
                OUTCOME_METRIC: float(np.clip(rng.normal(centre, 0.05), 0.0, 1.0)),
            })

    return pl.DataFrame(rows)


@pytest.fixture
def roles():
    """The resolved metric roles for a dataset from the reference producer."""
    return resolve_metric_roles(metric_names())
