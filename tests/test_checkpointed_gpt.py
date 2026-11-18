"""End-to-end checkpoint replay must preserve dropout gradients and RNG."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel


def pair(*, dropout: float = 0.2):
    config = GPTConfig(
        vocab_size=11, block_size=5, n_embd=12, n_head=3, n_layer=2,
        dropout=dropout, attention_impl="sdpa",
    )
    torch.manual_seed(123)
    ordinary = GPTLanguageModel(config)
    checkpointed = GPTLanguageModel(config)
    checkpointed.load_state_dict(ordinary.state_dict(), strict=True)
    checkpointed.blocks.set_activation_checkpointing(True)
    return ordinary, checkpointed


def test_dropout_and_checkpoint_replay_match_full_gpt_gradients():
    ordinary, checkpointed = pair(dropout=0.2)
    ordinary.train()
    checkpointed.train()
    inputs = torch.tensor([[0, 1, 2, 3], [3, 4, 2, 0]])
    targets = torch.tensor([[1, 2, 3, 4], [4, 2, 0, 1]])

    torch.manual_seed(234)
    loss_a = ordinary.loss(inputs, targets)
    loss_a.backward()
    state_after_a = torch.get_rng_state().clone()

    torch.manual_seed(234)
    loss_b = checkpointed.loss(inputs, targets)
    loss_b.backward()
    state_after_b = torch.get_rng_state().clone()

    torch.testing.assert_close(loss_a, loss_b, atol=1e-5, rtol=1e-5)
    for (name_a, a), (name_b, b) in zip(
        ordinary.named_parameters(), checkpointed.named_parameters()
    ):
        assert name_a == name_b
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-5)
    assert torch.equal(state_after_a, state_after_b)


def test_checkpointed_eval_matches_inference_and_preserves_tensor_keys():
    ordinary, checkpointed = pair(dropout=0.15)
    ordinary.eval()
    checkpointed.eval()
    tokens = torch.tensor([[0, 1, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(
            ordinary(tokens), checkpointed(tokens), atol=0, rtol=0,
        )
    assert list(ordinary.state_dict()) == list(checkpointed.state_dict())


def test_checkpointing_can_be_switched_off_without_replacing_weights():
    original, model = pair(dropout=0.0)
    model.blocks.set_activation_checkpointing(False)
    tokens = torch.tensor([[0, 1, 2]])
    with torch.no_grad():
        torch.testing.assert_close(original(tokens), model(tokens))
