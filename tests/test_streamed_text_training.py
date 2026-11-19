"""The real text trainer should hand the streamer CPU logical batches."""

import pytest
import torch

from rebuild_gpt.text_training import train_char_gpt


def corpus(tmp_path):
    path = tmp_path / "tiny.txt"
    path.write_text("ab" * 160, encoding="utf-8")
    return path


def opts():
    return dict(
        steps=3, block_size=3, batch_size=5, n_embd=8, n_head=2,
        n_layer=1, seed=143, learning_rate=0.001,
        eval_interval=1, eval_batches=1, attention_impl="sdpa",
    )


def test_logical_batch_reaches_streamer_on_cpu(tmp_path, monkeypatch):
    import rebuild_gpt.text_training as trainer

    original = trainer.train_streamed_logical_batch_step
    observed = []

    def spy(model, optimizer, inputs, targets, **kwargs):
        observed.append((inputs.device.type, targets.device.type,
                         inputs.size(0), kwargs["microbatch_size"]))
        return original(model, optimizer, inputs, targets, **kwargs)

    monkeypatch.setattr(trainer, "train_streamed_logical_batch_step", spy)
    train_char_gpt(corpus(tmp_path), microbatch_size=2, **opts())
    assert observed == [("cpu", "cpu", 5, 2)] * 3


@pytest.mark.parametrize("micro", [1, 2, 3, 5])
def test_streamed_and_unstreamed_loss_and_weights_are_close(tmp_path, micro):
    path = corpus(tmp_path)
    full, _, full_scores = train_char_gpt(path, **opts())
    streamed, _, stream_scores = train_char_gpt(
        path, microbatch_size=micro, **opts(),
    )
    for name in ("train_loss_last", "validation_before", "validation_after"):
        assert stream_scores[name] == pytest.approx(full_scores[name], abs=3e-5)
    for a, b in zip(full.parameters(), streamed.parameters()):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-5)


def test_streamed_checkpoint_resume_matches_uninterrupted(tmp_path):
    path = corpus(tmp_path)
    state = tmp_path / "resume.pt"
    options = opts()
    options["steps"] = 2
    train_char_gpt(
        path, microbatch_size=2, training_state_path=state, **options,
    )
    options["steps"] = 4
    resumed, _, resumed_scores = train_char_gpt(
        path, microbatch_size=2, resume_from=state, **options,
    )
    full, _, full_scores = train_char_gpt(
        path, microbatch_size=2, **options,
    )
    assert resumed_scores == full_scores
    assert all(torch.equal(a, b)
               for a, b in zip(resumed.parameters(), full.parameters()))
