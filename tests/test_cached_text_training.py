"""The cached sampler must preserve character-GPT learning and restarts."""

import json

import pytest
import torch

from rebuild_gpt.text_training import train_char_gpt
from rebuild_gpt.train_cli import main as train_main


def corpus(tmp_path):
    path = tmp_path / "story.txt"
    path.write_text("ab" * 140, encoding="utf-8")
    return path


def opts():
    return dict(
        steps=3, batch_size=5, block_size=3, n_embd=8,
        n_head=2, n_layer=1, seed=158,
        eval_interval=1, eval_batches=1, attention_impl="sdpa",
    )


@pytest.mark.parametrize("extra", [
    {},
    {"microbatch_size": 2},
    {"gradient_accumulation_steps": 2},
])
def test_cached_sampling_matches_original_training_bitwise(tmp_path, extra):
    path = corpus(tmp_path)
    baseline_hist, cached_hist = [], []
    a, _, a_scores = train_char_gpt(
        path, history=baseline_hist, sampling_backend="indexed",
        **opts(), **extra,
    )
    b, _, b_scores = train_char_gpt(
        path, history=cached_hist, sampling_backend="cached",
        **opts(), **extra,
    )
    assert a_scores == b_scores
    assert baseline_hist == cached_hist
    for left, right in zip(a.state_dict().values(), b.state_dict().values()):
        assert torch.equal(left, right)


def test_resume_rejects_switching_sampler_and_matches_continuation(tmp_path):
    path = corpus(tmp_path)
    saved = tmp_path / "state.pt"
    first = opts()
    first["steps"] = 2
    train_char_gpt(path, sampling_backend="cached",
                   training_state_path=saved, **first)
    later = opts()
    later["steps"] = 4
    with pytest.raises(ValueError, match="controls"):
        train_char_gpt(path, resume_from=saved, **later)
    resumed, _, resumed_scores = train_char_gpt(
        path, sampling_backend="cached", resume_from=saved, **later)
    full, _, full_scores = train_char_gpt(
        path, sampling_backend="cached", **later)
    assert resumed_scores == full_scores
    assert all(torch.equal(a, b)
               for a, b in zip(resumed.parameters(), full.parameters()))


def test_invalid_sampler_policy_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="sampling_backend"):
        train_char_gpt(corpus(tmp_path), sampling_backend="unknown", **opts())


def test_cli_records_opt_in_backend(tmp_path):
    path = corpus(tmp_path)
    report = tmp_path / "report.json"
    args = [
        "--corpus", str(path), "--steps", "1", "--batch-size", "3",
        "--block-size", "3", "--n-embd", "8", "--n-head", "2",
        "--n-layer", "1", "--eval-every", "1", "--eval-batches", "1",
        "--sampling-backend", "cached",
        "--output", str(tmp_path / "model.pt"),
        "--best-output", str(tmp_path / "best.pt"),
        "--history-file", str(report),
    ]
    assert train_main(args) == 0
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["training_controls"]["sampling_backend"] == "cached"
