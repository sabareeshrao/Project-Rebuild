"""A syntactically valid 'passed' report may still exceed a local budget."""

from unittest.mock import patch

import pytest

from rebuild_gpt.conformance_budgets import check_conformance_budgets


def _valid_checks():
    names = {
        "causal_prefix": {"max_prefix_error": 0.0},
        "batch_independence": {"max_batch_error": 1e-6},
        "forward_contract": {
            "max_train_logit_error": 0.0,
            "max_last_logit_error": 0.0,
            "masked_loss_error": 0.0,
        },
        "attention_backends": {
            "max_logit_error": 1e-6, "loss_error": 1e-6,
            "max_gradient_error": 1e-6,
        },
        "finite_difference": {"absolute_error": 1e-5},
    }
    return {"checks": names}


def test_eight_numerical_measurements_within_limits():
    with patch(
        "rebuild_gpt.conformance_budgets.load_conformance_report",
        return_value=_valid_checks(),
    ):
        result = check_conformance_budgets("audit.json")
    assert result["status"] == "within_budget"
    assert result["checked_measurements"] == 9


def test_oversized_gradient_error_fails_even_when_report_is_well_formed():
    report = _valid_checks()
    report["checks"]["finite_difference"]["absolute_error"] = 0.03
    with patch(
        "rebuild_gpt.conformance_budgets.load_conformance_report",
        return_value=report,
    ):
        with pytest.raises(AssertionError, match="finite_difference.absolute_error"):
            check_conformance_budgets("audit.json")


def test_oversized_attention_error_is_identified_by_field():
    report = _valid_checks()
    report["checks"]["attention_backends"]["max_gradient_error"] = 0.1
    with patch(
        "rebuild_gpt.conformance_budgets.load_conformance_report",
        return_value=report,
    ):
        with pytest.raises(AssertionError, match="attention_backends.max_gradient_error"):
            check_conformance_budgets("audit.json")


@pytest.mark.parametrize("settings", [
    {"max_logit_error": -1},
    {"max_logit_error": float("nan")},
    {"max_logit_error": True},
    {"max_derivative_error": float("inf")},
])
def test_invalid_budget_rejected_before_opening_report(settings):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        check_conformance_budgets("missing.json", **settings)
