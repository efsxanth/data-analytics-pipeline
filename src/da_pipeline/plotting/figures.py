"""
figures.py -- the three separate concerns behind "plotting settings".

1. Theme      -- how figures look. analytics.mplstyle, applied per figure.
2. Geometry   -- size, DPI, output location. A per-run FigureConfig.
3. Encoding   -- which colour means which condition. An explicit, validated map.

There is no set_style(). The theme is applied inside a plt.rc_context that
spans figure creation through saving, so:

  * importing this module changes nothing about the process;
  * a caller's rcParams are left exactly as they were;
  * it works inside a ProcessPoolExecutor worker, where a style applied once
    at the entry point would not survive (child processes do not re-run the
    __main__ guard under spawn, which is what Windows uses).

That last point is the reason for the design. Figures are generated in worker
processes, so the style has to travel with the figure, not with the process.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ..config import PipelineConfig

THEME_PATH = Path(__file__).with_name("analytics.mplstyle")


@lru_cache(maxsize=1)
def theme() -> dict:
    """The theme as an rcParams dict, read from disk once per process.

    use_default_template=False means only the keys named in the file are
    returned, so applying it overrides those and leaves everything else alone.
    """
    if not THEME_PATH.exists():
        raise FileNotFoundError(
            f"Theme file not found: {THEME_PATH}. It must sit beside "
            f"{Path(__file__).name}."
        )
    return matplotlib.rc_params_from_file(THEME_PATH, use_default_template=False)


def default_colours() -> list[str]:
    """The theme's colour cycle, as a plain list.

    The palette is defined once, in the style sheet, rather than duplicated in
    the argument parser or the config defaults.
    """
    cycle = theme().get("axes.prop_cycle")
    if cycle is None:
        raise KeyError(f"axes.prop_cycle is not set in {THEME_PATH.name}.")
    return list(cycle.by_key()["color"])


def resolve_palette(config: PipelineConfig) -> list[str]:
    """The colours this run should use: the user's, or the theme's.

    ``PlotSettings.colours`` is None unless the user named colours explicitly.
    Resolving it here, rather than in the config, is what keeps matplotlib off
    the import path of anything that merely wants to describe a run.
    """
    if config.plots.colours:
        return list(config.plots.colours)
    return default_colours()


# ---------------------------------------------------------------------------
# Geometry -- a per-run parameter, not global state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FigureConfig:
    """Size and destination for this run's figures.

    Frozen because it is read from worker processes: nothing should be able to
    change it halfway through a set of figures.

    DPI lives here rather than in the theme on purpose. As an rcParam it would
    silently apply to every figure in the process, including ones this pipeline
    did not create.
    """

    width: float
    height: float
    dpi: int
    output_dir: str
    fmt: str = "png"

    @classmethod
    def from_config(cls, config: PipelineConfig) -> FigureConfig:
        """Extract the geometry a figure needs from the run's config."""
        return cls(
            width=config.plots.figure_width,
            height=config.plots.figure_height,
            dpi=config.plots.dpi,
            output_dir=str(config.resolved_output_dir),
        )

    def figsize(self, height_scale: float = 1.0) -> tuple[float, float]:
        return self.width, self.height * height_scale


# ---------------------------------------------------------------------------
# Encoding -- colour is data, so it is built once and validated
# ---------------------------------------------------------------------------

def build_condition_colours(
    conditions: Sequence[str],
    palette: Sequence[str],
) -> dict[str, str]:
    """Map each condition to a fixed colour.

    Built once, up front, and passed to every plot. Keying by name rather than
    by position means a condition cannot end up a different colour in two
    figures of the same run, and a missing entry fails here rather than
    rendering in some arbitrary default several figures later.

    Raises:
        ValueError: if there are more conditions than colours, or duplicates.
    """
    if len(set(conditions)) != len(conditions):
        duplicates = sorted({c for c in conditions if list(conditions).count(c) > 1})
        raise ValueError(f"Duplicate conditions: {duplicates}")

    if len(conditions) > len(palette):
        raise ValueError(
            f"{len(conditions)} condition(s) but only {len(palette)} colour(s) "
            f"available. Pass more via --colours."
        )

    return dict(zip(conditions, palette))


# ---------------------------------------------------------------------------
# Figure lifecycle -- create, style, save and close in one place
# ---------------------------------------------------------------------------

@contextmanager
def figure(
    config: FigureConfig,
    filename: str,
    *subdirs: str,
    nrows: int = 1,
    height_ratios: list | None = None,
    height_scale: float = 1.0,
    sharex: bool = False,
    sharey: bool = False,
    tight_layout_rect: list | None = None,
) -> Iterator[tuple]:
    """Yield (fig, axes), then save under output_dir/<subdirs>/<filename>.

        with figure(config, 'snr.png', 'signal_to_noise') as (fig, ax):
            sns.boxplot(ax=ax, ...)
            ax.set_title(...)

    The theme is active for the whole block, including the save: savefig reads
    rcParams at save time, so styling only the creation would lose half of it.

    The figure is closed on the way out even if the body raises. At 300 dpi a
    leaked figure is several megabytes, and a run producing dozens of them will
    exhaust memory and trip matplotlib's open-figure warning.

    For nrows > 1, axes is always an array, so callers never special-case the
    single-panel shape.
    """
    with plt.rc_context(theme()):
        fig, axes = plt.subplots(
            nrows,
            1,
            sharex=sharex,
            sharey=sharey,
            figsize=config.figsize(height_scale),
            gridspec_kw={"height_ratios": height_ratios} if height_ratios else None,
        )

        if nrows > 1:
            axes = np.atleast_1d(axes)

        try:
            yield fig, axes

            output_path = os.path.join(config.output_dir, *subdirs)
            os.makedirs(output_path, exist_ok=True)

            if tight_layout_rect:
                fig.tight_layout(rect=tight_layout_rect)
            else:
                fig.tight_layout()

            fig.savefig(os.path.join(output_path, filename), dpi=config.dpi)
        finally:
            plt.close(fig)
