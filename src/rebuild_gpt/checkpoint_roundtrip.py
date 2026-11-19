"""Independently reload a saved GPT through the public inference bundle API."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from rebuild_gpt.embedding_tie_check import check_embedding_tying
from rebuild_gpt.model_invariants import _validate_token_batch
from rebuild_gpt.text_checkpoint import load_text_checkpoint, save_text_checkpoint


def check_checkpoint_roundtrip(
    checkpoint_path: str | Path,
    tokens: torch.Tensor,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict:
    """Save->reload via a temporary file; compare state and predictions.

    The input checkpoint is never overwritten. Saving and reloading
    instantiate models, so restore the caller's global CPU RNG afterward.
    """
    prior_rng = torch.get_rng_state().clone()
    try:
        before, tokenizer = load_text_checkpoint(checkpoint_path)
        _validate_token_batch(before, tokens)
        if tokens.device.type != "cpu":
            raise ValueError("The round-trip check currently requires CPU tokens")
        if not before.config.tie_embeddings:
            raise ValueError("The nanoGPT round-trip check requires tied embeddings")
        check_embedding_tying(before)
        with TemporaryDirectory(prefix="gpt-roundtrip-") as folder:
            temporary = Path(folder) / "model.pt"
            save_text_checkpoint(before, tokenizer, temporary)
            after, after_tokenizer = load_text_checkpoint(temporary)
            check_embedding_tying(after)
            if tokenizer.vocab != after_tokenizer.vocab or before.config != after.config:
                raise AssertionError("Checkpoint round trip changed tokenizer or config")
            old_state, new_state = before.state_dict(), after.state_dict()
            if set(old_state) != set(new_state):
                raise AssertionError("Checkpoint round trip changed model state keys")
            for key in sorted(old_state):
                if not torch.equal(old_state[key], new_state[key]):
                    raise AssertionError(f"Checkpoint round trip changed tensor: {key}")
            with torch.no_grad():
                expected = before(tokens)
                actual = after(tokens)
            try:
                torch.testing.assert_close(expected, actual, atol=atol, rtol=rtol)
            except AssertionError as exc:
                raise AssertionError("Checkpoint round trip changed model logits") from exc
        return {
            "identical": True,
            "state_tensors": len(old_state),
            "vocab_size": before.config.vocab_size,
            "checked_logits": expected.numel(),
        }
    finally:
        torch.set_rng_state(prior_rng)
