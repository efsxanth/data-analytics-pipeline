"""What still predicts the outcome once the predictors are considered together.

The correlation tables answer one predictor at a time. That cannot separate
predictors that move with each other: signal-to-noise correlates with the
outcome, a degradation metric correlates with the outcome, and signal-to-noise
correlates with that metric -- so at least one of the first two relationships may
be nothing more than a reflection of the third. A regression carrying every
predictor at once reports what each still contributes with the others held
constant.

Which metric is the outcome and which are predictors is the run's configuration,
resolved into a :class:`~da_pipeline.metrics.MetricRoles`. Nothing here names a
metric.

Three decisions shape everything below.

**Fitted inside each run.** For exactly the reason the correlations are: pooling
observations from several runs mixes how observations covary inside a run with how run
averages covary across runs, and the mixture can inflate a coefficient, shrink
it, or reverse its sign. Each run is fitted on its own and the coefficients are
then summarised across runs, so the number of runs -- the real replicate count
-- is what supports any claim of reproducibility.

**Condition is not a predictor.** Not a choice: condition is constant within a
run, so a per-run design matrix containing it would be singular. The question
"does the condition change the outcome?" is answered by the pairwise comparisons
and the variance decomposition, on run medians. The question here is a different
one -- given a run, which observation properties go with a better-scoring
observation.

**Ordinary least squares on the raw outcome.** An outcome that is a bounded
proportion usually argues for a logit or binomial model. It is not done here,
for two reasons. The extracts carry every metric as a bare float with no
numerator or denominator, so a binomial model is not available at all. And the
inference reported below comes from resampling *runs*, not from the regression's
own standard errors -- so the assumptions those standard errors rest on, constant
residual variance chief among them, do the little work that would make a
transform worth its interpretive cost. Each per-run fit is a descriptive summary
of within-run association; the run-level bootstrap does the inferential work.

What the output supports is an **association within a run**. "Observations with
more signal tend to score better" is what a coefficient says. It is not evidence
that raising signal-to-noise on a given observation would raise its score, and
nothing here is a mediation analysis.

Two things are left out on purpose. The outcome's **component metrics** are not
predictors: each is another description of the same measured outcome, so
regressing the outcome on them would mostly recover the arithmetic relating them
to it. That is why the resolved roles keep them apart, and their relationship to
the outcome is covered by the correlation tables and the bin summaries instead.
And when a run contains several **groups**, their observations are fitted
together; groups sit at different absolute levels, so a small amount of the same
between-group covariation this design removes at run level survives at group
level.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from ..constants import columns as col
from .inference import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_N_RESAMPLES,
    bootstrap_ci_1sample,
)

logger = logging.getLogger(__name__)

# A run needs enough observations that the coefficients are not fitted to noise.
# Well below anything real -- runs here carry thousands -- so it is a guard
# against a degenerate input, not a modelling threshold.
MIN_OBSERVATIONS_PER_RUN = 30

# Leave-one-run-out needs runs to leave in as well as one to leave out.
MIN_RUNS_FOR_CROSS_VALIDATION = 3

# A predictor whose within-run spread is this small is constant for practical
# purposes; standardising it would divide by ~0 and hand the fit a column of
# arbitrarily large numbers.
MIN_STANDARD_DEVIATION = 1e-12


def standardise(values: np.ndarray) -> np.ndarray | None:
    """Z-score a column, or None if it has no spread to score by."""

    spread = float(values.std())
    if not np.isfinite(spread) or spread < MIN_STANDARD_DEVIATION:
        return None
    return (values - values.mean()) / spread


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Share of the outcome's variance the predictions account for.

    Written against the *actual* mean rather than taken from a fit's residuals,
    so the same function serves in-sample and out-of-sample use. Out of sample
    the value can go negative, which is informative rather than broken: it means
    the coefficients predicted the held-out run worse than that run's own mean
    would have.
    """

    total = float(((actual - actual.mean()) ** 2).sum())
    if total <= 0.0:
        return float("nan")

    residual = float(((actual - predicted) ** 2).sum())
    return 1.0 - residual / total


class OutcomeModeller:
    """Per-run multivariable regression of the outcome metric on its predictors."""

    def __init__(
        self,
        predictors: list[str],
        outcome: str,
        *,
        n_resamples: int = DEFAULT_N_RESAMPLES,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    ):
        """
        Args:
            predictors: The metrics to carry as predictors, alongside anything
                else the caller adds such as the signal-to-noise ratio.
            outcome: The metric being predicted.
            n_resamples: Bootstrap resamples, over runs.
            confidence_level: Interval width for the bootstrap.
        """
        self.predictors = list(predictors)
        self.outcome = outcome
        self.n_resamples = n_resamples
        self.confidence_level = confidence_level

    # ------------------------------------------------------------------
    # One run
    # ------------------------------------------------------------------
    def _design_matrix(self, run_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
        """Standardised (X, y) for one run, or None if the run cannot be fitted.

        Listwise deletion, unlike the pairwise deletion the correlations use: a
        multivariable coefficient is only defined on rows where every predictor
        is present, so a row missing one of them cannot contribute to any of
        them. Non-finite values are dropped here too -- signal-to-noise is inf
        by design when an observation's level 0 median is zero.
        """

        columns = [self.outcome, *self.predictors]
        raw = run_df.select(columns).to_numpy().astype(float)

        complete = raw[np.isfinite(raw).all(axis=1)]
        if complete.shape[0] < max(MIN_OBSERVATIONS_PER_RUN, len(self.predictors) + 2):
            return None

        standardised = [standardise(complete[:, i]) for i in range(complete.shape[1])]

        # One flat column and the whole run is dropped, rather than silently
        # fitting a smaller model: coefficients from two different predictor
        # sets are not the same quantity and must not be summarised together.
        if any(column is None for column in standardised):
            return None

        matrix = np.column_stack(standardised)
        return matrix[:, 1:], matrix[:, 0]

    @staticmethod
    def _fit(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
        """Least-squares coefficients, or None if the predictors are collinear.

        No intercept column: both sides were centred by standardisation, so the
        intercept is zero by construction. A rank-deficient matrix means two
        predictors carry the same information and the individual coefficients
        are arbitrary, so the run is dropped rather than reported.
        """

        coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        if rank < x.shape[1]:
            return None
        return coefficients

    def _variance_inflation(self, x: np.ndarray) -> list[float]:
        """Per-predictor VIF: 1 / (1 - R^2) of that predictor on the others.

        A VIF of 5 means the coefficient's standard error is sqrt(5) times what
        it would be if the predictor were independent of the rest -- the number
        is still an unbiased estimate, just a much less stable one.
        """

        inflations = []

        for index in range(x.shape[1]):
            others = np.delete(x, index, axis=1)
            this_predictor = x[:, index]

            if others.shape[1] == 0:
                inflations.append(1.0)
                continue

            coefficients, _, rank, _ = np.linalg.lstsq(others, this_predictor, rcond=None)
            if rank < others.shape[1]:
                inflations.append(float("inf"))
                continue

            explained = r_squared(this_predictor, others @ coefficients)
            inflations.append(
                float("inf") if explained >= 1.0 else 1.0 / (1.0 - explained)
            )

        return inflations

    # ------------------------------------------------------------------
    # Every run
    # ------------------------------------------------------------------
    def _fit_every_run(self, matrix_df: pl.DataFrame) -> list[dict]:
        """One fitted model per run, in a stable order."""

        fits = []

        for (condition, run_id), run_df in matrix_df.group_by(
            [col.CONDITION, col.RUN_ID], maintain_order=True
        ):
            prepared = self._design_matrix(run_df)
            if prepared is None:
                logger.warning(
                    f"Skipping run {run_id}: too few complete observation rows, or a "
                    f"predictor with no spread within the run."
                )
                continue

            x, y = prepared
            coefficients = self._fit(x, y)
            if coefficients is None:
                logger.warning(
                    f"Skipping run {run_id}: the predictors are collinear within "
                    f"this run, so the individual coefficients are not identified."
                )
                continue

            fits.append({
                col.CONDITION: condition,
                col.RUN_ID: run_id,
                col.N_OBSERVATIONS: int(y.size),
                "coefficients": coefficients,
                "vif": self._variance_inflation(x),
                "r_squared": r_squared(y, x @ coefficients),
                "x": x,
                "y": y,
            })

        return sorted(fits, key=lambda fit: (fit[col.CONDITION], fit[col.RUN_ID]))

    # ------------------------------------------------------------------
    # Leave-one-run-out cross-validation
    # ------------------------------------------------------------------
    def _cross_validate(self, fits: list[dict]) -> list[float]:
        """Out-of-sample R^2 for each run, fitted on all the others.

        Sign agreement across runs measures consistency: whether the runs tell
        the same story. It cannot measure whether that story predicts a run the
        fit has never seen, and with correlated predictors the two come apart --
        coefficients can agree in sign run after run while carrying no
        transferable information about any individual observation.

        Holding a run out is what closes that gap, and standardising within each
        run is what makes it possible. Runs sit at different absolute levels of
        the outcome and of every predictor, so raw coefficients from one run cannot
        be applied to another; z-scores put every run on the same scale, and
        with both sides centred there is no intercept to transfer either. The
        median coefficient vector across the training runs is used rather than a
        refit on their pooled rows, because pooling would reintroduce exactly
        the between-run covariation this design exists to keep out.
        """

        if len(fits) < MIN_RUNS_FOR_CROSS_VALIDATION:
            logger.info(
                f"Leave-one-run-out cross-validation needs at least "
                f"{MIN_RUNS_FOR_CROSS_VALIDATION} runs; {len(fits)} were fitted."
            )
            return [float("nan")] * len(fits)

        scores = []

        for held_out, fit in enumerate(fits):
            training = np.array([
                other["coefficients"]
                for index, other in enumerate(fits)
                if index != held_out
            ])
            scores.append(
                r_squared(fit["y"], fit["x"] @ np.median(training, axis=0))
            )

        return scores

    # ------------------------------------------------------------------
    # Output tables
    # ------------------------------------------------------------------
    def build_model_tables(
        self, matrix_df: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Fit every run and summarise.

        Args:
            matrix_df: The wide per-observation matrix -- signal-to-noise and
                every analysed metric on one row per observation, as built by
                ``DataCorrelator.build_per_observation_matrix``.

        Returns:
            (summary, by_run). The summary carries one row per predictor with
            its coefficient across runs and how well the model generalises; the
            per-run table carries one row per (run, predictor) so the summary
            can be checked and any single run's contribution seen.
        """

        missing = [
            name for name in (self.outcome, *self.predictors)
            if name not in matrix_df.columns
        ]
        if missing:
            logger.warning(
                f"{self.outcome} model: skipped, columns not present: {missing}."
            )
            return pl.DataFrame(), pl.DataFrame()

        fits = self._fit_every_run(matrix_df)

        if not fits:
            logger.warning(f"{self.outcome} model: no run could be fitted.")
            return pl.DataFrame(), pl.DataFrame()

        out_of_sample = self._cross_validate(fits)

        by_run_df = self._build_by_run_df(fits, out_of_sample)
        summary_df = self._build_summary_df(fits, out_of_sample)

        logger.info(f"- {self.outcome} model: {summary_df.shape} over {len(fits)} runs")

        return summary_df, by_run_df

    def _build_by_run_df(
        self, fits: list[dict], out_of_sample: list[float]
    ) -> pl.DataFrame:
        """One row per (run, predictor): the detail behind the summary."""

        rows = [
            {
                col.CONDITION: fit[col.CONDITION],
                col.RUN_ID: fit[col.RUN_ID],
                col.OUTCOME: self.outcome,
                col.PREDICTOR: predictor,
                col.COEFFICIENT: float(fit["coefficients"][index]),
                col.VIF: _finite_or_none(fit["vif"][index]),
                col.R_SQUARED_IN_SAMPLE: _finite_or_none(fit["r_squared"]),
                col.R_SQUARED_OUT_OF_SAMPLE: _finite_or_none(score),
                col.N_OBSERVATIONS: fit[col.N_OBSERVATIONS],
                col.STANDARDISATION: col.STANDARDISED_WITHIN_RUN,
            }
            for fit, score in zip(fits, out_of_sample)
            for index, predictor in enumerate(self.predictors)
        ]

        return pl.DataFrame(rows)

    def _build_summary_df(
        self, fits: list[dict], out_of_sample: list[float]
    ) -> pl.DataFrame:
        """One row per predictor, plus the model-level fit repeated on each.

        The generalisation columns are the same on every row, which is
        deliberate: they answer "is this model worth reading at all?", and a
        reader who opens the file at one predictor should not have to look
        elsewhere to find out.
        """

        in_sample = np.array([fit["r_squared"] for fit in fits], dtype=float)
        held_out = np.array(out_of_sample, dtype=float)
        usable_held_out = held_out[np.isfinite(held_out)]

        model_level = {
            col.N_RUNS_MODELLED: len(fits),
            col.R_SQUARED_IN_SAMPLE_MEDIAN: _finite_or_none(_median(in_sample)),
            col.R_SQUARED_OUT_OF_SAMPLE_MEDIAN: _finite_or_none(_median(held_out)),
            col.N_RUNS_PREDICTED: int((usable_held_out > 0).sum()),
            col.CI_LEVEL: self.confidence_level,
            col.CI_METHOD: col.CI_METHOD_BOOTSTRAP_RUNS,
            col.N_BOOTSTRAP: self.n_resamples,
            col.STANDARDISATION: col.STANDARDISED_WITHIN_RUN,
            # Each coefficient is fitted over the observations of one run; only the
            # median, range and sign count across coefficients are at run level.
            col.UNIT_OF_ANALYSIS: col.UNIT_OBSERVATION_WITHIN_RUN,
        }

        rows = []

        for index, predictor in enumerate(self.predictors):
            coefficients = np.array(
                [fit["coefficients"][index] for fit in fits], dtype=float
            )
            inflations = np.array([fit["vif"][index] for fit in fits], dtype=float)

            ci_low, ci_high = bootstrap_ci_1sample(
                coefficients,
                np.median,
                n_resamples=self.n_resamples,
                confidence_level=self.confidence_level,
            )

            rows.append({
                col.OUTCOME: self.outcome,
                col.PREDICTOR: predictor,
                col.COEFFICIENT_MEDIAN: float(np.median(coefficients)),
                col.COEFFICIENT_MIN: float(coefficients.min()),
                col.COEFFICIENT_MAX: float(coefficients.max()),
                col.COEFFICIENT_CI_LOW: ci_low,
                col.COEFFICIENT_CI_HIGH: ci_high,
                # The plainest statement of reproducibility available at twenty
                # runs, and the one a reader can check against the per-run file.
                col.COEFFICIENT_SAME_SIGN: int(
                    max((coefficients > 0).sum(), (coefficients < 0).sum())
                ),
                col.VIF_MEDIAN: _finite_or_none(_median(inflations)),
                col.VIF_MAX: _finite_or_none(float(inflations.max())),
                **model_level,
            })

        return pl.DataFrame(rows)


def _median(values: np.ndarray) -> float:
    """Median over the finite entries, NaN if there are none."""
    usable = values[np.isfinite(values)]
    return float(np.median(usable)) if usable.size else float("nan")


def _finite_or_none(value: float) -> float | None:
    """NaN and inf become null in the CSV rather than the strings 'NaN'/'inf'."""
    return float(value) if np.isfinite(value) else None
