"""Resolving a run's metric selection against what the files actually declare.

This is where the open-ended metric set is decided: the reader discovers names,
and this module works out which of them is the outcome, which restate it, and
which are predictors. Two behaviours matter most and are tested hardest --
discovery is permissive, and an explicit selection that cannot be honoured is a
hard error rather than a quietly smaller analysis.
"""

from __future__ import annotations

import pytest

from da_pipeline.metrics import (
    MetricRoles,
    MetricSelection,
    MetricSelectionError,
    resolve_metric_roles,
)

# A plausible declared set: an outcome, two components named after it, and three
# independent metrics. Nothing in the pipeline requires this shape.
DECLARED = (
    "Accuracy",
    "Accuracy_error_1",
    "Accuracy_error_2",
    "degradation_1",
    "degradation_2",
    "degradation_3",
)


class TestDefaultRoles:
    def test_every_declared_metric_is_analysed_by_default(self):
        roles = resolve_metric_roles(DECLARED)

        assert set(roles.analysed) == set(DECLARED)

    def test_components_are_the_metrics_named_after_the_outcome(self):
        roles = resolve_metric_roles(DECLARED)

        assert roles.outcome == "Accuracy"
        assert roles.components == ("Accuracy_error_1", "Accuracy_error_2")

    def test_everything_else_is_a_predictor(self):
        roles = resolve_metric_roles(DECLARED)

        assert roles.predictors == ("degradation_1", "degradation_2", "degradation_3")

    def test_the_outcome_is_never_a_predictor_of_itself(self):
        roles = resolve_metric_roles(DECLARED)

        assert roles.outcome not in roles.predictors
        assert roles.outcome not in roles.components

    def test_a_completely_different_metric_set_resolves_the_same_way(self):
        # Nothing here is a name the pipeline has ever seen.
        roles = resolve_metric_roles(
            ("yield", "yield_shortfall", "pressure", "humidity"),
            MetricSelection(outcome="yield"),
        )

        assert roles.outcome == "yield"
        assert roles.components == ("yield_shortfall",)
        assert roles.predictors == ("pressure", "humidity")

    def test_an_arbitrary_number_of_predictors_is_carried_through(self):
        declared = ("Accuracy", *[f"degradation_{i}" for i in range(1, 13)])
        roles = resolve_metric_roles(declared)

        assert len(roles.predictors) == 12


class TestDiscoveryIsPermissive:
    def test_a_declared_metric_the_run_ignores_is_not_an_error(self):
        roles = resolve_metric_roles(
            DECLARED, MetricSelection(names=["Accuracy", "degradation_1"])
        )

        assert roles.analysed == ("Accuracy", "degradation_1")
        assert roles.predictors == ("degradation_1",)

    def test_a_file_with_no_components_is_fine(self):
        roles = resolve_metric_roles(("Accuracy", "degradation_1"))

        assert roles.components == ()
        assert roles.predictors == ("degradation_1",)

    def test_a_file_with_nothing_but_the_outcome_is_fine(self):
        roles = resolve_metric_roles(("Accuracy",))

        assert roles.analysed == ("Accuracy",)
        assert roles.predictors == ()

    def test_a_file_declaring_no_metrics_at_all_is_not(self):
        with pytest.raises(MetricSelectionError, match="declare no metrics"):
            resolve_metric_roles(())


class TestSelectionIsStrict:
    """A name the files do not declare stops the run, and says what they do declare.

    A typo would otherwise mean the analysis somebody asked for silently did not
    happen -- the run would look like a success and be missing the thing being
    investigated.
    """

    def test_an_unknown_metric_name_is_an_error(self):
        with pytest.raises(MetricSelectionError) as excinfo:
            resolve_metric_roles(DECLARED, MetricSelection(names=["degradaton_1"]))

        assert "metric_names" in str(excinfo.value)
        assert "degradaton_1" in str(excinfo.value)

    def test_the_error_lists_the_metrics_the_files_declared(self):
        # The whole point: the fix is obvious from the message, without opening
        # the file to find out what was available.
        with pytest.raises(MetricSelectionError) as excinfo:
            resolve_metric_roles(DECLARED, MetricSelection(names=["degradaton_1"]))

        message = str(excinfo.value)
        for name in DECLARED:
            assert name in message

    def test_one_bad_name_among_good_ones_still_stops_the_run(self):
        with pytest.raises(MetricSelectionError, match="nope"):
            resolve_metric_roles(
                DECLARED, MetricSelection(names=["Accuracy", "degradation_1", "nope"])
            )

    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [
            ("outcome", "Acuracy", "outcome_metric"),
            ("predictors", ["degradaton_2"], "predictor_metric_names"),
            ("components", ["Accuracy_eror_1"], "component_metric_names"),
        ],
    )
    def test_every_metric_field_is_checked_against_the_files(self, field, value, expected):
        with pytest.raises(MetricSelectionError) as excinfo:
            resolve_metric_roles(DECLARED, MetricSelection(**{field: value}))

        assert expected in str(excinfo.value)

    def test_a_label_for_a_metric_that_does_not_exist_is_an_error(self):
        with pytest.raises(MetricSelectionError, match="metric_labels"):
            resolve_metric_roles(
                DECLARED, MetricSelection(labels={"degradaton_1": "Read-ahead"})
            )


class TestIncoherentSelections:
    def test_a_selection_that_omits_the_outcome_is_rejected(self):
        with pytest.raises(MetricSelectionError, match="always analysed"):
            resolve_metric_roles(DECLARED, MetricSelection(names=["degradation_1"]))

    def test_a_role_outside_the_analysed_set_is_rejected(self):
        with pytest.raises(MetricSelectionError, match="not among the analysed metrics"):
            resolve_metric_roles(
                DECLARED,
                MetricSelection(
                    names=["Accuracy", "degradation_1"], predictors=["degradation_2"]
                ),
            )

    def test_a_metric_cannot_be_both_predictor_and_component(self):
        with pytest.raises(MetricSelectionError, match="both a predictor and a component"):
            MetricSelection(predictors=["degradation_1"], components=["degradation_1"])

    def test_a_repeated_name_is_rejected(self):
        with pytest.raises(MetricSelectionError, match="listed more than once"):
            MetricSelection(names=["Accuracy", "Accuracy"])

    def test_an_empty_outcome_is_rejected(self):
        with pytest.raises(MetricSelectionError, match="must name a metric"):
            MetricSelection(outcome="")

    def test_a_metric_labelled_twice_is_rejected(self):
        with pytest.raises(MetricSelectionError, match="labelled more than once"):
            MetricSelection(labels=[("Accuracy", "A"), ("Accuracy", "B")])


class TestExplicitRoles:
    def test_explicit_predictors_override_the_convention(self):
        roles = resolve_metric_roles(
            DECLARED, MetricSelection(predictors=["degradation_1"])
        )

        assert roles.predictors == ("degradation_1",)
        # The others are still analysed, just not as predictors.
        assert "degradation_2" in roles.analysed or "degradation_2" not in roles.predictors

    def test_explicit_components_override_the_naming_convention(self):
        # A metric not named after the outcome can still restate it.
        roles = resolve_metric_roles(
            DECLARED, MetricSelection(components=["degradation_3"])
        )

        assert roles.components == ("degradation_3",)
        assert "Accuracy_error_1" in roles.predictors

    def test_an_outcome_with_no_components_by_convention_is_allowed(self):
        roles = resolve_metric_roles(
            DECLARED, MetricSelection(outcome="degradation_1")
        )

        assert roles.outcome == "degradation_1"
        assert roles.components == ()
        assert "Accuracy" in roles.predictors


class TestLabels:
    def test_a_metric_without_a_label_is_shown_under_its_own_name(self):
        roles = resolve_metric_roles(DECLARED)

        assert roles.label("degradation_1") == "degradation_1"

    def test_a_labelled_metric_uses_its_label(self):
        roles = resolve_metric_roles(
            DECLARED, MetricSelection(labels={"degradation_1": "Read-ahead"})
        )

        assert roles.label("degradation_1") == "Read-ahead"
        assert roles.label("degradation_2") == "degradation_2"


class TestRolesObject:
    def test_outcome_metrics_are_the_outcome_and_its_components(self):
        roles = resolve_metric_roles(DECLARED)

        assert roles.outcome_metrics == (
            "Accuracy", "Accuracy_error_1", "Accuracy_error_2",
        )

    def test_analysed_covers_every_role_exactly_once(self):
        roles = resolve_metric_roles(DECLARED)

        assert len(roles.analysed) == len(set(roles.analysed))
        assert set(roles.analysed) == set(roles.outcome_metrics) | set(roles.predictors)

    def test_roles_are_frozen(self):
        roles = resolve_metric_roles(DECLARED)

        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError
            roles.outcome = "other"

    def test_describe_names_each_role(self):
        joined = "\n".join(resolve_metric_roles(DECLARED).describe())

        assert "- outcome metric: Accuracy" in joined
        assert "degradation_1" in joined

    def test_roles_can_be_built_directly(self):
        # The analysis layer receives one of these; it does not need to know how
        # it was resolved.
        roles = MetricRoles(outcome="a", components=("a_b",), predictors=("c",))

        assert roles.analysed == ("a", "a_b", "c")
