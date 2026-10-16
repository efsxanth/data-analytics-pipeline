"""An offline analytics pipeline for repeated-measurement extract files.

Reads a set of HDF5 extracts grouped into experimental conditions, derives
per-observation quality metrics from them, and writes a directory of figures and
CSVs comparing the conditions. The extract format is this project's own; see
:mod:`da_pipeline.constants.schema` for its specification and
:mod:`da_pipeline.synthetic` for a producer of it.

Typical use is the ``da-pipeline`` command, but the same run can be described
in Python:

    >>> from pathlib import Path
    >>> from da_pipeline import PipelineConfig, run
    >>> config = PipelineConfig(
    ...     config_path=Path("runs.json"),
    ...     response_name="auc",
    ...     group_ids=[3],
    ... )
    >>> run(config)                                   # doctest: +SKIP

``run`` is resolved lazily so that importing this package -- to build a config,
or to reach for the extract reader -- does not pull in matplotlib.
"""

from .config import (
    ConfigError,
    ExportSelection,
    PipelineConfig,
    PlotSettings,
    StepRange,
)
from .metrics import MetricRoles, MetricSelection, MetricSelectionError

__version__ = "1.0.0"

__all__ = [
    "ConfigError",
    "ExportSelection",
    "MetricRoles",
    "MetricSelection",
    "MetricSelectionError",
    "PipelineConfig",
    "PlotSettings",
    "StepRange",
    "run",
    "__version__",
]


def __getattr__(name: str):
    """Resolve ``run`` on first use (PEP 562), keeping plotting imports off the import path."""
    if name == "run":
        from .pipeline import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
