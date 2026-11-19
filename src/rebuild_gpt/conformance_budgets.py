"""Apply explicit numerical error limits to a saved GPT conformance report."""

import math
from pathlib import Path

from rebuild_gpt.conformance_report import load_conformance_report


ERROR_MEASUREMENTS = {
    "causal_prefix": ("max_prefix_error",),
    "batch_independence": ("max_batch_error",),
    "forward_contract": (
        "max_train_logit_error", "max_last_logit_error", "masked_loss_error",
    ),
    "attention_backends": ("max_logit_error", "loss_error", "max_gradient_error"),
    "finite_difference": ("absolute_error",),
}


def check_conformance_budgets(
    report_path: str | Path,
    *,
    max_logit_error: float = 2e-5,
    max_derivative_error: float = 1e-3,
) -> dict:
    """Reject numerical drifts even if a JSON report claims it passed.

    Budgets are explicit *absolute* limits, not cryptographic or model
    quality guarantees. The finite-difference gradient gets its own
    limit because numerical differentiation is approximate.
    """
    for label, limit in (
        ("max_logit_error", max_logit_error),
        ("max_derivative_error", max_derivative_error),
    ):
        if (type(limit) not in (float, int)
                or not math.isfinite(limit) or limit < 0):
            raise ValueError(f"{label} must be finite and nonnegative")
    report = load_conformance_report(report_path)
    observations = {}
    for group, fields in ERROR_MEASUREMENTS.items():
        for field in fields:
            value = report["checks"][group][field]
            limit = (
                max_derivative_error if group == "finite_difference"
                else max_logit_error
            )
            if value > limit:
                raise AssertionError(
                    f"Numerical drift exceeds budget: {group}.{field} = {value} > {limit}"
                )
            observations[f"{group}.{field}"] = {
                "observed": value, "limit": limit,
            }
    return {
        "status": "within_budget",
        "checked_measurements": len(observations),
        "measurements": observations,
    }
