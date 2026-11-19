"""CUDA staging options must be explicit and not disturb CPU defaults."""

import json

import pytest
import torch

from rebuild_gpt.text_training import train_char_gpt
from rebuild_gpt.train_cli import main as train_main


def make_corpus(tmp_path):
    path = tmp_path / "tiny.txt"
    path.write_text("ab" * 120, encoding="utf-8")
    return path


def settings():
    return dict(
        steps=2, block_size=3, n_embd=8, n_head=2, n_layer=1,
        batch_size=5, seed=150, eval_interval=1, eval_batches=1,
        attention_impl="sdpa", microbatch_size=2,
    )


def test_default_streaming_still_works_and_matches_explicit_disabled_flags(tmp_path):
    path = make_corpus(tmp_path)
    a, _, metrics_a = train_char_gpt(path, **settings())
    b, _, metrics_b = train_char_gpt(
        path, pin_memory=False, non_blocking_transfer=False, **settings(),
    )
    assert metrics_a == metrics_b
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))


def test_cuda_flags_are_rejected_on_cpu_before_optimizer_initializes(tmp_path):
    path = make_corpus(tmp_path)
    with pytest.raises(ValueError, match="pin_memory requires CUDA"):
        train_char_gpt(path, pin_memory=True, **settings())
    with pytest.raises(ValueError, match="requires pin_memory"):
        train_char_gpt(path, non_blocking_transfer=True, **settings())


def test_transfer_mode_requires_logical_microbatch_mode(tmp_path):
    path = make_corpus(tmp_path)
    options = settings()
    options.pop("microbatch_size")
    with pytest.raises(ValueError, match="require microbatch_size"):
        train_char_gpt(path, pin_memory=True, **options)


def test_cli_records_off_state_and_rejects_unsupported_cpu_mode(tmp_path):
    corpus = make_corpus(tmp_path)
    report = tmp_path / "history.json"
    args = [
        "--corpus", str(corpus), "--steps", "1",
        "--block-size", "3", "--n-embd", "8", "--n-head", "2",
        "--n-layer", "1", "--batch-size", "5", "--microbatch-size", "2",
        "--eval-every", "1", "--eval-batches", "1",
        "--output", str(tmp_path / "final.pt"),
        "--best-output", str(tmp_path / "best.pt"),
        "--history-file", str(report),
    ]
    assert train_main(args) == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["training_controls"]["pin_memory"] is False
    assert data["training_controls"]["non_blocking_transfer"] is False
    with pytest.raises(SystemExit) as error:
        train_main(args + ["--pin-memory"])
    assert error.value.code == 2


def test_cli_rejects_missing_prerequisites():
    with pytest.raises(SystemExit) as first:
        train_main(["--pin-memory"])
    assert first.value.code == 2
    with pytest.raises(SystemExit) as second:
        train_main([
            "--microbatch-size", "2", "--non-blocking-transfer",
        ])
    assert second.value.code == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_pinning_flags_preserve_learning_and_resume_compatibility(tmp_path):
    path = make_corpus(tmp_path)
    ordinary, _, score_a = train_char_gpt(
        path, device="cuda", **settings(),
    )
    staged, _, score_b = train_char_gpt(
        path, device="cuda", pin_memory=True,
        non_blocking_transfer=True, **settings(),
    )
    assert score_a["validation_after"] == pytest.approx(
        score_b["validation_after"], abs=1e-4
    )
    for a, b in zip(ordinary.parameters(), staged.parameters()):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    checkpoint = tmp_path / "cuda_state.pt"
    train_char_gpt(
        path, device="cuda", pin_memory=True,
        non_blocking_transfer=True,
        training_state_path=checkpoint, **settings(),
    )
    later = settings()
    later["steps"] = 4
    with pytest.raises(ValueError, match="controls"):
        train_char_gpt(
            path, device="cuda", resume_from=checkpoint, **later,
        )
    resumed, _, _ = train_char_gpt(
        path, device="cuda", pin_memory=True,
        non_blocking_transfer=True,
        resume_from=checkpoint, **later,
    )
    assert next(resumed.parameters()).device.type == "cuda"
