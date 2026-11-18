from dataclasses import dataclass

import pytest
import torch

from rebuild_gpt.text_resume import load_text_training_state, save_text_training_state


@dataclass
class ToyConfig:
    vocab_size: int = 2


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = ToyConfig()
        self.linear = torch.nn.Linear(2, 2)


class ToyTokenizer:
    vocab = ("a", "b")
    vocab_size = 2


def test_resume_state_round_trip_and_setup_checks(tmp_path):
    corpus = tmp_path / "text.txt"
    corpus.write_text("abababa", encoding="utf-8")
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    gen = torch.Generator().manual_seed(17)
    torch.randint(0, 10, (5,), generator=gen)
    path = tmp_path / "training.pt"
    options = {"batch_size": 2, "seed": 17}
    save_text_training_state(
        path, model=model, tokenizer=ToyTokenizer(), optimizer=optimizer,
        corpus_path=corpus, controls=options, step=4, batch_generator=gen,
        history=[{"step": 4}], validation_before=1.2,
        last_loss=0.8, last_learning_rate=0.02,
    )
    saved = load_text_training_state(
        path, model=model, tokenizer=ToyTokenizer(),
        corpus_path=corpus, controls=options,
    )
    assert saved["step"] == 4
    assert saved["history"] == [{"step": 4}]
    other = torch.Generator()
    other.set_state(saved["batch_generator_state"])
    assert torch.equal(
        torch.randint(0, 10, (7,), generator=gen),
        torch.randint(0, 10, (7,), generator=other),
    )
    with pytest.raises(ValueError, match="controls"):
        load_text_training_state(
            path, model=model, tokenizer=ToyTokenizer(),
            corpus_path=corpus, controls={"seed": 4},
        )
    corpus.write_text("other data", encoding="utf-8")
    with pytest.raises(ValueError, match="corpus_sha256"):
        load_text_training_state(
            path, model=model, tokenizer=ToyTokenizer(),
            corpus_path=corpus, controls=options,
        )


def test_failed_save_does_not_destroy_last_good_resume_point(tmp_path, monkeypatch):
    corpus = tmp_path / "text.txt"
    corpus.write_text("ab" * 30, encoding="utf-8")
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    generator = torch.Generator().manual_seed(8)
    output = tmp_path / "train_state.pt"
    kwargs = dict(
        model=model, tokenizer=ToyTokenizer(), optimizer=optimizer,
        corpus_path=corpus, controls={"seed": 8}, step=2,
        batch_generator=generator, history=[], validation_before=1.0,
        last_loss=0.7, last_learning_rate=0.02,
    )
    save_text_training_state(output, **kwargs)
    known_good = output.read_bytes()

    def fail_to_save(*args, **kwargs):
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(torch, "save", fail_to_save)
    with pytest.raises(OSError, match="interrupted"):
        save_text_training_state(output, **kwargs)
    assert output.read_bytes() == known_good
    assert not list(tmp_path.glob(".train_state.pt.*.tmp"))
