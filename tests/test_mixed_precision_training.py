"""BF16 training is available on CPU, FP16 requires CUDA hardware."""

import pytest
import torch

from rebuild_gpt.text_training import train_char_gpt


def corpus(tmp_path):
    path = tmp_path / "training.txt"
    path.write_text("ab" * 120, encoding="utf-8")
    return path


@pytest.mark.parametrize("accumulation", [1, 3])
def test_cpu_bf16_training_and_evaluation(tmp_path, accumulation):
    model, tok, metrics = train_char_gpt(
        corpus(tmp_path), steps=3, block_size=3, n_embd=8,
        n_head=2, n_layer=1, batch_size=4, eval_interval=1,
        eval_batches=2, gradient_accumulation_steps=accumulation,
        precision="bf16", device="cpu", max_grad_norm=1.0,
    )
    assert next(model.parameters()).dtype == torch.float32
    assert tok.vocab_size == 2
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()


def test_bf16_training_can_save_and_resume_state(tmp_path):
    path = corpus(tmp_path)
    state = tmp_path / "bf16.pt"
    opts = dict(
        block_size=3, n_embd=8, n_head=2, n_layer=1,
        batch_size=3, precision="bf16", eval_interval=2, eval_batches=1, seed=31,
    )
    train_char_gpt(path, steps=2, training_state_path=state, **opts)
    resumed, _, _ = train_char_gpt(path, steps=4, resume_from=state, **opts)
    continuous, _, _ = train_char_gpt(path, steps=4, **opts)
    assert all(torch.equal(x, y) for x, y in zip(resumed.parameters(), continuous.parameters()))


def test_fp16_rejected_on_cpu_before_training(tmp_path):
    with pytest.raises(ValueError, match="requires CUDA"):
        train_char_gpt(corpus(tmp_path), steps=1, precision="fp16", device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_or_fp16_training(tmp_path):
    mode = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    model, _, results = train_char_gpt(
        corpus(tmp_path), steps=2, block_size=3, n_embd=8,
        n_head=2, n_layer=1, batch_size=3,
        precision=mode, device="cuda", eval_interval=1,
    )
    assert next(model.parameters()).device.type == "cuda"
    assert results["train_loss_last"] > 0
