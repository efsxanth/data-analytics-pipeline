"""The run's derived statistics, packaged for a language model.

Same content as the report, shaped for a different reader. Three decisions
define it.

**Derived statistics only.** No filesystem path, no configuration, no
per-observation row ever enters the pack. What goes in is what the analysis concluded, which is
also all that is needed to interpret it.

**Run identifiers are pseudonymised**, but only when that changes something. A
pack is the thing most likely to be pasted somewhere else, so it should not be
the thing carrying an organisation's internal run naming: an identifier like
``machine4_20240115b`` is replaced by ``run_08``, and the mapping back is written
to a separate file that stays on the machine. Where the identifiers are already
in the canonical ``run_NN`` form there is nothing to hide and nothing to decode,
so they are used unchanged and no mapping file is written -- an identity mapping
is a file that answers a question nobody can ask.

Condition names are kept either way: the whole analysis is a comparison between
conditions, and an interpretation that comes back discussing ``condition_2`` has
to be decoded by hand before it is any use. A deployment that considers its
condition names sensitive can rename them in its own configuration, where the
change flows through every output consistently.

**Floats are rounded.** Six significant figures is far beyond what a handful of
runs can support, and full double precision roughly doubles the file for digits
that are noise.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

import polars as pl

from ..constants import columns as col
from ..constants import outputs as out
from .findings import RunFindings

logger = logging.getLogger(__name__)

AI_SUBDIR = 'ai'
PACK_FNAME = 'evidence_pack.json'
RUN_MAPPING_FNAME = 'run_id_mapping.json'

# Beyond this the digits are an artefact of double precision, not a measurement.
SIGNIFICANT_FIGURES = 6

# Per-observation exports are inputs to the analysis, not results of it. They
# are also where the raw data actually lives, so they never enter the pack.
EXCLUDED_FROM_PACK = frozenset({
    out.RESPONSES_CSV_FNAME,
    out.SNR_CSV_FNAME,
    out.PREDICTOR_METRICS_CSV_FNAME,
    out.OUTCOME_METRICS_CSV_FNAME,
})

# Columns known to carry a run identifier. Anything matching is pseudonymised.
RUN_ID_COLUMNS = (col.RUN_ID, col.MOST_INFLUENTIAL_RUN)


def _round(value):
    """A float trimmed to significant figures, or the value untouched."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        if value == 0.0:
            return 0.0
        digits = SIGNIFICANT_FIGURES - int(math.floor(math.log10(abs(value)))) - 1
        return round(value, max(digits, 0))
    return value


def build_run_mapping(run_ids: list[str]) -> dict[str, str]:
    """Original run identifier -> ``run_01``, ``run_02``, ...

    Sorted naturally, so ``run-2`` precedes ``run-10`` rather than following
    it: the numbering is meant to be readable next to the original, and a
    lexicographic order would scramble that for anyone checking the mapping.
    """

    def natural_key(value: str):
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', value)
        ]

    return {
        run_id: f"run_{index:02d}"
        for index, run_id in enumerate(sorted(run_ids, key=natural_key), start=1)
    }


def renames_anything(mapping: dict[str, str]) -> bool:
    """Whether this mapping actually changes any identifier.

    A dataset whose runs are already named ``run_01``, ``run_02``, ... maps onto
    itself. Nothing is concealed by applying it and nothing can be recovered from
    writing it down, so both steps are skipped rather than performed emptily.
    """

    return any(run_id != alias for run_id, alias in mapping.items())


def collect_run_ids(frames: list[pl.DataFrame]) -> list[str]:
    """Every distinct run identifier appearing anywhere in the analysis output."""

    found: set[str] = set()
    for frame in frames:
        if frame is None or frame.is_empty():
            continue
        for column in RUN_ID_COLUMNS:
            if column in frame.columns:
                found.update(
                    value for value in frame[column].unique().to_list()
                    if value is not None
                )
    return sorted(found)


def pseudonymise(frame: pl.DataFrame, mapping: dict[str, str]) -> pl.DataFrame:
    """Replace run identifiers wherever they appear in a frame.

    An identifier with no entry in the mapping is left as it is rather than
    silently nulled: an unmapped value means the collection missed a column, and
    a visible oddity in the pack is easier to notice than a hole.
    """

    columns = [c for c in RUN_ID_COLUMNS if c in frame.columns]
    if not columns:
        return frame

    return frame.with_columns([
        pl.col(c).replace(mapping).alias(c) for c in columns
    ])


def pseudonymise_text(text: str | None, mapping: dict[str, str]) -> str | None:
    """Replace run identifiers inside a sentence.

    The findings are written from the in-memory frames, before any
    pseudonymisation, because the report they also feed is local and should name
    the run a reader can go and look at. The pack is the copy that travels, so
    the substitution happens here instead -- and it has to work on prose, not
    just on columns, because a finding like "dropping run-13 moves the mean"
    carries the identifier in the middle of a sentence.

    Longest identifier first, or ``run-1`` would rewrite the front of
    ``run-13`` and leave a ``run_01 3`` behind.
    """

    if not text or not mapping:
        return text

    ordered = sorted(mapping, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(run_id) for run_id in ordered))

    return pattern.sub(lambda match: mapping[match.group(0)], text)


def _records(frame: pl.DataFrame, mapping: dict[str, str]) -> list[dict]:
    return [
        {key: _round(value) for key, value in row.items()}
        for row in pseudonymise(frame, mapping).iter_rows(named=True)
    ]


def merge_bin_summaries(paths: list[Path]) -> pl.DataFrame | None:
    """Rejoin the per-metric bin summaries into the one frame they came from.

    On disk they are split one file per metric, and every file repeats the bin
    identity and the whole outcome summary beside its own metric -- which is
    right for a CSV a person opens on its own, and wasteful here. Nine files
    carrying fourteen identical columns each accounted for roughly two thirds of
    an early pack, for information already present eight times over.

    Rejoining costs nothing and loses nothing: the shared columns appear once,
    each metric keeps its own, and the result is the frame the saver split.
    """

    if not paths:
        return None

    frames = []
    for path in paths:
        try:
            frames.append(pl.read_csv(path))
        except Exception:                          # noqa: BLE001 - one bad file must not lose the rest
            logger.warning(f"Evidence pack: could not read {path.name}, table skipped.")

    if not frames:
        return None

    keys = [col.CONDITION, col.OUTCOME_BIN_INDEX]
    if not all(set(keys) <= set(frame.columns) for frame in frames):
        # An unexpected shape: keep the files separate rather than join on a
        # key that may not identify a row.
        logger.warning("Evidence pack: bin summaries lack a shared key, left unmerged.")
        return None

    merged = frames[0]
    for frame in frames[1:]:
        new_columns = [c for c in frame.columns if c not in merged.columns]
        if new_columns:
            merged = merged.join(frame.select(keys + new_columns), on=keys, how="left")

    return merged


def collect_tables(output_dir: Path, mapping: dict[str, str]) -> dict:
    """Every analysis table from the run, pseudonymised and rounded.

    Read back from the CSVs rather than taken from the in-memory frames, so the
    pack is guaranteed to describe what the run actually wrote. If the two ever
    disagreed, the file on disk is the one somebody would check.
    """

    csv_root = output_dir / out.CSV_EXPORTS_SUBDIR
    tables: dict = {}

    if not csv_root.is_dir():
        return tables

    bin_summary_paths = sorted((csv_root / out.BIN_SUMMARY_SUBDIR).glob("*.csv"))

    for path in sorted(csv_root.rglob("*.csv")):
        if path.name in EXCLUDED_FROM_PACK or path in bin_summary_paths:
            continue
        try:
            frame = pl.read_csv(path)
        except Exception:                          # noqa: BLE001 - one bad file must not lose the pack
            logger.warning(f"Evidence pack: could not read {path.name}, table skipped.")
            continue

        relative = path.relative_to(csv_root)
        if relative.parent == Path("."):
            tables[path.stem] = _records(frame, mapping)
        else:
            group = str(relative.parent).replace("\\", "/")
            tables.setdefault(group, {})[path.stem] = _records(frame, mapping)

    merged_bins = merge_bin_summaries(bin_summary_paths)
    if merged_bins is not None:
        tables[out.BIN_SUMMARY_SUBDIR] = _records(merged_bins, mapping)

    return tables


def build_evidence_pack(
    findings: RunFindings,
    output_dir: Path,
    mapping: dict[str, str],
) -> dict:
    """The whole pack: what the run found, and the numbers behind it."""

    return {
        "schema": "da-pipeline/evidence-pack/2",
        "design": findings.design,
        "findings": [
            {
                key: pseudonymise_text(value, mapping)
                for key, value in finding.to_dict().items()
            }
            for finding in findings.findings
        ],
        "tables": collect_tables(output_dir, mapping),
        "notes": {
            "run_identifiers": (
                "Pseudonymised. The mapping back to the original identifiers was "
                "not included in this pack."
                if renames_anything(mapping)
                else "Already in the canonical run_NN form; used unchanged."
            ),
            "unit_of_analysis": findings.design.get("unit_of_analysis"),
            "rounding": (
                f"Floats are rounded to {SIGNIFICANT_FIGURES} significant figures; "
                f"further digits are double-precision artefacts, not measurement."
            ),
        },
    }


def write_evidence_pack(
    findings: RunFindings,
    output_dir: Path,
    analysis_frames: list[pl.DataFrame],
) -> Path:
    """Write the pack and the local-only mapping, and return the pack's path."""

    ai_dir = output_dir / AI_SUBDIR
    ai_dir.mkdir(parents=True, exist_ok=True)

    mapping = build_run_mapping(collect_run_ids(analysis_frames))

    pack_path = ai_dir / PACK_FNAME
    pack_path.write_text(
        json.dumps(build_evidence_pack(findings, output_dir, mapping), indent=2),
        encoding="utf-8",
    )

    # Written beside the pack rather than inside it, and named so its purpose is
    # obvious to anyone about to share the directory. Skipped entirely when the
    # identifiers were already canonical: the file would map every name to
    # itself, which tells a reader nothing and still looks like a secret.
    if renames_anything(mapping):
        (ai_dir / RUN_MAPPING_FNAME).write_text(
            json.dumps(
                {
                    "purpose": (
                        "Maps the pseudonyms used in evidence_pack.json back to this "
                        "run's original identifiers. Kept out of the pack on purpose; "
                        "do not send this file with it."
                    ),
                    "mapping": {alias: run_id for run_id, alias in mapping.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        renamed = f"{len(mapping)} runs pseudonymised"
    else:
        renamed = f"{len(mapping)} runs already named run_NN"

    size_kb = pack_path.stat().st_size / 1024
    logger.info(f"- Evidence pack: {PACK_FNAME} ({size_kb:.0f} KB, {renamed})")

    return pack_path
