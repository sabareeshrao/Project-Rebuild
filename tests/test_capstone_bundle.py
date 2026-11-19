"""Checkpoint bundling keeps the tokenizer, model and sample together."""

import json

from rebuild_gpt.capstone import save_capstone_run
from rebuild_gpt.capstone_spec import CapstoneSpec
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.tokenizer import CharTokenizer


def test_capstone_bundle_contains_reloadable_model_and_sample(tmp_path):
    corpus = tmp_path / "story.txt"
    corpus.write_text("ab ab ab ab", encoding="utf-8")
    tokenizer = CharTokenizer(corpus.read_text(encoding="utf-8"))
    model = GPTLanguageModel(GPTConfig(tokenizer.vocab_size, 4, 8, 2, 1)).eval()
    spec = CapstoneSpec(new_tokens=4, prompt="a", seed=5)
    output = tmp_path / "run"

    report = save_capstone_run(
        model, tokenizer, {"validation_after": 1.4}, corpus, output, spec,
    )
    assert (output / "model.pt").is_file()
    assert (output / "sample.txt").read_text(encoding="utf-8") == report["generated_text"]
    assert report == json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert len(report["generated_text"]) == len(spec.prompt) + spec.new_tokens
    assert report["checkpoint"] == "model.pt"
    assert len(report["checkpoint_sha256"]) == 64
    assert len(report["corpus_sha256"]) == 64
