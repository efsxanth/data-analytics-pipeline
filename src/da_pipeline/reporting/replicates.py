"""Replicate counts, turned into the sentences the outputs state them in.

Three of this package's outputs -- the AI briefing, the findings' cautions and
the HTML report's reading rules -- argue from how many runs each condition has.
"a large p-value is not evidence of equivalence" is only convincing next to the
smallest p-value the test could have returned at *this* many runs, and "the
intervals are wide" only means something once the reader knows what they are
wide from.

Those sentences were written out by hand once, against the design that happened
to be in front of the author, and then stayed put while the data changed. The
result was a briefing whose rules quoted a larger and more even design than the
evidence pack beside them came from -- rules a reader could check against the
design summary two sections up and find false.

So the sentences are built here instead, from ``runs_per_condition`` as the run
actually reported it, and shared by all three consumers so they cannot drift
apart from each other either.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..analysis.pairwise import MIN_RUNS_PER_CONDITION, smallest_attainable_two_sided_p

# Prose reads better with small counts spelled out, and a design large enough to
# exhaust this list is large enough that digits stop looking out of place.
NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
}

# What every one of these sentences falls back to when the design is not known
# or holds nothing comparable. Vaguer than the generated version and deliberately
# so: a rule that cannot name the count is still worth stating.
GENERIC_PHRASE = "these replicate counts"


def spell(count: int) -> str:
    """A run count as prose: ``three`` up to twelve, then digits."""
    return NUMBER_WORDS.get(count, str(count))


def replicate_counts(runs_per_condition: Mapping[str, int] | None) -> list[int]:
    """The distinct per-condition run counts, smallest first."""
    if not runs_per_condition:
        return []
    return sorted({int(n) for n in runs_per_condition.values() if int(n) > 0})


def replicate_phrase(
    runs_per_condition: Mapping[str, int] | None,
    *,
    suffix: str = " runs per condition",
) -> str:
    """How many runs a condition has here, as a phrase a sentence can take.

    One count reads as ``three runs per condition``; several read as the range
    they span, ``two to three runs per condition``. A range is a bound rather
    than a list, which stays honest for a design whose counts are not contiguous.
    """

    counts = replicate_counts(runs_per_condition)

    if not counts:
        return GENERIC_PHRASE
    if len(counts) == 1:
        return f"{spell(counts[0])}{suffix}"

    return f"{spell(counts[0])} to {spell(counts[-1])}{suffix}"


def comparable_pairings(runs_per_condition: Mapping[str, int] | None) -> list[tuple[int, int]]:
    """Every distinct pair of group sizes the pairwise comparison will actually run.

    Sizes below :data:`MIN_RUNS_PER_CONDITION` are dropped, because those
    comparisons are skipped rather than reported, and a floor quoted for a
    comparison nobody will see is noise. Largest pair first, so the sentence
    built from these opens with the best case the design can manage.
    """

    counts = [n for n in replicate_counts(runs_per_condition) if n >= MIN_RUNS_PER_CONDITION]
    largest_first = sorted(counts, reverse=True)

    return [
        (a, b)
        for index, a in enumerate(largest_first)
        for b in largest_first[index:]
    ]


def _fmt_p(p: float) -> str:
    """A p-value floor at the precision that distinguishes it from its neighbours."""
    return f"{p:.3g}"


def p_floor_clause(runs_per_condition: Mapping[str, int] | None) -> str:
    """The smallest two-sided p each comparison in this design could return.

    Complete separation -- every run of one condition above every run of the
    other -- is the most extreme result the data can produce, so the p-value it
    yields is a floor. Stating it per pair of group sizes is the whole point:
    with unequal conditions the floor differs between comparisons in the same
    table, and a reader who knows only the best case will misread the rest.
    """

    pairings = comparable_pairings(runs_per_condition)

    if not pairings:
        return (
            "the smallest attainable two-sided Mann-Whitney p-value is bounded "
            "away from zero"
        )

    # "runs" is carried by the first clause only; repeating it in every clause
    # turns a list of three floors into a paragraph.
    parts = [
        f"{_fmt_p(smallest_attainable_two_sided_p(a, b))} for {spell(a)}"
        f"{' runs' if index == 0 else ''} against {spell(b)}"
        for index, (a, b) in enumerate(pairings)
    ]

    joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f", and {parts[-1]}"

    return f"the smallest attainable two-sided Mann-Whitney p-value is {joined}"


def p_floor_span_clause(runs_per_condition: Mapping[str, int] | None) -> str:
    """The same floors as :func:`p_floor_clause`, for prose with room for two numbers.

    A caution attached to a finding is read in passing, so it gets the span --
    the best comparison this design can manage and the worst -- rather than the
    floor for every pair of sizes. Both ends matter: the best case is what a
    reader assumes applies everywhere, and the worst is the one that makes a
    large p-value meaningless.
    """

    pairings = comparable_pairings(runs_per_condition)

    if not pairings:
        return "the smallest attainable two-sided p-value is bounded away from zero"

    best = _fmt_p(smallest_attainable_two_sided_p(*pairings[0]))

    if len(pairings) == 1:
        return f"the smallest attainable two-sided p-value is {best}"

    worst = _fmt_p(smallest_attainable_two_sided_p(*pairings[-1]))

    return (
        f"the smallest attainable two-sided p-value runs from {best} for the "
        f"largest comparison here to {worst} for the smallest"
    )
