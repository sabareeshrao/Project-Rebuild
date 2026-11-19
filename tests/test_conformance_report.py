"""A JSON success label alone cannot constitute GPT conformance evidence."""

from copy import deepcopy
import json

import pytest

from rebuild_gpt.conformance_report import load_conformance_report


def _example():
    return {
        "format_version": 1, "status": "passed",
        "settings": {"batch_size": 2, "seed": 13},
        "model": {"vocab_size": 5, "block_size": 4, "n_layer": 1},
        "sha256": {"checkpoint": "a" * 64, "corpus": "b" * 64},
        "checks": {
            "causal_prefix": {"checked_positions": 6, "max_prefix_error": 0.0},
            "batch_independence": {"checked_samples": 2, "max_batch_error": 0.0},
            "forward_contract": {
                "supervised_positions": 6, "max_train_logit_error": 0.0,
                "max_last_logit_error": 0.0, "masked_loss_error": 0.0,
            },
            "finite_difference": {"absolute_error": 1e-10},
            "embedding_tying": {"tied": True, "embedding_elements": 40, "vocab_size": 5},
            "seeded_generation": {
                "repeatable": True, "global_rng_preserved": True, "seed": 13,
            },
            "checkpoint_roundtrip": {"identical": True, "state_tensors": 9},
            "attention_backends": {
                "matched_parameter_gradients": 9, "max_logit_error": 0.0,
                "loss_error": 0.0, "max_gradient_error": 0.0,
            },
        },
    }


def _load(tmp_path, obj):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(obj), encoding="utf-8")
    return load_conformance_report(path)


def test_valid_report_round_trips_without_modification(tmp_path):
    sample = _example()
    assert _load(tmp_path, sample) == sample


@pytest.mark.parametrize("change", [
    lambda r: r.update(status="failed"),
    lambda r: r["checks"].pop("causal_prefix"),
    lambda r: r["checks"]["embedding_tying"].update(tied=False),
    lambda r: r["checks"]["seeded_generation"].update(global_rng_preserved=False),
    lambda r: r["checks"]["seeded_generation"].update(seed=99),
    lambda r: r["checks"]["forward_contract"].update(masked_loss_error=-1),
    lambda r: r["checks"]["finite_difference"].update(absolute_error=float("nan")),
    lambda r: r["checks"]["attention_backends"].update(max_gradient_error="zero"),
    lambda r: r["sha256"].update(checkpoint="A" * 64),
    lambda r: r["model"].update(vocab_size=9),
    lambda r: r["settings"].update(batch_size=True),
])
def test_malformed_or_false_claims_are_rejected(tmp_path, change):
    candidate = deepcopy(_example())
    change(candidate)
    with pytest.raises(ValueError):
        _load(tmp_path, candidate)
