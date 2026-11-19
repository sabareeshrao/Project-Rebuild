"""Validate claims in an eight-check GPT conformance JSON report.

A report's own 'passed' status is only a *claim*. Schema validation is
the first step; later lessons independently inspect the source files.
"""

import json
import math
from pathlib import Path
import re


CHECK_NAMES = frozenset({
    "causal_prefix", "batch_independence", "forward_contract",
    "finite_difference", "embedding_tying", "seeded_generation",
    "checkpoint_roundtrip", "attention_backends",
})
ERROR_FIELDS = {
    "causal_prefix": ("max_prefix_error",),
    "batch_independence": ("max_batch_error",),
    "forward_contract": (
        "max_train_logit_error", "max_last_logit_error", "masked_loss_error",
    ),
    "finite_difference": ("absolute_error",),
    "attention_backends": ("max_logit_error", "loss_error", "max_gradient_error"),
}
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def validate_conformance_report(report: dict) -> dict:
    """Reject missing, malformed, or contradictory success claims."""
    if not isinstance(report, dict) or report.get("format_version") != 1:
        raise ValueError("Unsupported conformance report version")
    if report.get("status") != "passed":
        raise ValueError("Conformance report does not say passed")
    checks = report.get("checks")
    if not isinstance(checks, dict) or set(checks) != CHECK_NAMES:
        raise ValueError("Conformance report must contain exactly eight named checks")
    if any(not isinstance(check, dict) for check in checks.values()):
        raise ValueError("Conformance checks must be JSON objects")
    for check_name, fields in ERROR_FIELDS.items():
        for field in fields:
            value = checks[check_name].get(field)
            if (type(value) not in (float, int) or not math.isfinite(value)
                    or value < 0):
                raise ValueError(f"Invalid conformance error: {check_name}.{field}")
    for name, field in (
        ("embedding_tying", "tied"),
        ("seeded_generation", "repeatable"),
        ("seeded_generation", "global_rng_preserved"),
        ("checkpoint_roundtrip", "identical"),
    ):
        if checks[name].get(field) is not True:
            raise ValueError(f"Conformance check failed: {name}.{field}")
    for check_name, field in (
        ("causal_prefix", "checked_positions"),
        ("batch_independence", "checked_samples"),
        ("forward_contract", "supervised_positions"),
        ("embedding_tying", "embedding_elements"),
        ("checkpoint_roundtrip", "state_tensors"),
        ("attention_backends", "matched_parameter_gradients"),
    ):
        value = checks[check_name].get(field)
        if type(value) is not int or value < 1:
            raise ValueError(f"Invalid count: {check_name}.{field}")
    config = report.get("model")
    settings = report.get("settings")
    if not isinstance(config, dict) or not isinstance(settings, dict):
        raise ValueError("Conformance report is missing model or run settings")
    for field in ("vocab_size", "block_size", "n_layer"):
        value = config.get(field)
        if type(value) is not int or value < 1:
            raise ValueError(f"Invalid model {field}")
    for field in ("seed", "batch_size"):
        value = settings.get(field)
        if type(value) is not int or value < (0 if field == "seed" else 2):
            raise ValueError(f"Invalid conformance {field}")
    if checks["embedding_tying"].get("vocab_size") != config["vocab_size"]:
        raise ValueError("Reported embedding and model vocabulary differ")
    if checks["seeded_generation"].get("seed") != settings["seed"]:
        raise ValueError("Seeded generation used a different seed")
    hashes = report.get("sha256")
    if not isinstance(hashes, dict):
        raise ValueError("Conformance report is missing file hashes")
    for field in ("checkpoint", "corpus"):
        digest = hashes.get(field)
        if not isinstance(digest, str) or _HEX.fullmatch(digest) is None:
            raise ValueError(f"Invalid SHA-256 digest for {field}")
    return report


def load_conformance_report(path: str | Path) -> dict:
    """Parse and structurally validate a saved conformance JSON file."""
    return validate_conformance_report(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
