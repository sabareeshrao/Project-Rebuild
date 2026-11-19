"""Check that the capstone reuses—not reimplements—the GPT trainer."""

from unittest.mock import patch

from rebuild_gpt.capstone import train_capstone
from rebuild_gpt.capstone_spec import CapstoneSpec


def test_capstone_forwards_only_training_settings():
    spec = CapstoneSpec(steps=2, prompt="b", new_tokens=5, seed=11)
    expected = ("model", "tokenizer", {"validation_after": 1.2})
    with patch("rebuild_gpt.capstone.train_char_gpt", return_value=expected) as trainer:
        actual = train_capstone("data/tiny_story.txt", spec)
    assert actual == expected
    trainer.assert_called_once_with("data/tiny_story.txt", **spec.training_kwargs())
