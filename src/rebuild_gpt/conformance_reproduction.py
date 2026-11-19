"""Independently rerun all eight checks recorded in a trusted GPT audit."""

import math
from pathlib import Path

from rebuild_gpt.conformance_integrity import verify_conformance_integrity
from rebuild_gpt.conformance_report import load_conformance_report
from rebuild_gpt.extended_conformance import run_extended_conformance


def _compare_measurements(expected, actual, path: str,
                          *, atol: float, rtol: float) -> int:
    """Compare nested data, allowing tiny floating-point rounding changes."""
    if type(expected) is not type(actual):
        raise AssertionError(f"{path}: value types differ")
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            raise AssertionError(f"{path}: result fields differ")
        return sum(
            _compare_measurements(expected[k], actual[k], f"{path}.{k}",
                                  atol=atol, rtol=rtol)
            for k in sorted(expected)
        )
    if isinstance(expected, list):
        if len(expected) != len(actual):
            raise AssertionError(f"{path}: list lengths differ")
        return sum(
            _compare_measurements(a, b, f"{path}[{i}]",
                                  atol=atol, rtol=rtol)
            for i, (a, b) in enumerate(zip(expected, actual))
        )
    if isinstance(expected, float):
        if not math.isclose(expected, actual, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(f"{path}: saved measurement differs from a new run")
    elif expected != actual:
        raise AssertionError(f"{path}: saved value differs from a new run")
    return 1


def reproduce_conformance_report(
    report_path: str | Path,
    checkpoint_path: str | Path,
    corpus_path: str | Path,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> dict:
    """Recompute all checks from hash-verified files; never overwrite report.

    Only load locally trusted checkpoint data. Changing the checkpoint,
    corpus or a reported measurement must not silently produce success.
    """
    if (type(atol) not in (float, int) or type(rtol) not in (float, int)
            or not math.isfinite(atol) or not math.isfinite(rtol)
            or atol < 0 or rtol < 0):
        raise ValueError("Comparison tolerances must be finite and nonnegative")
    verified = verify_conformance_integrity(
        report_path, checkpoint_path, corpus_path,
    )
    saved = load_conformance_report(report_path)
    settings = saved["settings"]
    fresh = run_extended_conformance(
        checkpoint_path, corpus_path,
        batch_size=settings["batch_size"], seed=settings["seed"],
    )
    for key in ("format_version", "status", "settings", "sha256", "model", "environment"):
        if saved.get(key) != fresh.get(key):
            raise AssertionError(f"conformance.{key}: saved metadata differs from a new run")
    matched = _compare_measurements(
        saved["checks"], fresh["checks"], "checks", atol=atol, rtol=rtol,
    )
    return {
        "status": "reproduced",
        "reproduced_checks": len(saved["checks"]),
        "compared_measurements": matched,
        "checkpoint_sha256": verified["checkpoint_sha256"],
        "corpus_sha256": verified["corpus_sha256"],
    }
