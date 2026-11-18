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


def test_cpu_checkpoint_does_not_initialize_or_require_cuda(tmp_path):
    corpus = tmp_path / "cpu.txt"
    corpus.write_text("abcabc", encoding="utf-8")
    model = ToyModel()
    opt = torch.optim.AdamW(model.parameters())
    output = tmp_path / "cpu.pt"
    save_text_training_state(
        output, model=model, tokenizer=ToyTokenizer(), optimizer=opt,
        corpus_path=corpus, controls={"device": "cpu"}, step=0,
        batch_generator=torch.Generator().manual_seed(1),
        history=[], validation_before=1.0, last_loss=1.0, last_learning_rate=0.01,
    )
    state = load_text_training_state(
        output, model=model, tokenizer=ToyTokenizer(), corpus_path=corpus,
        controls={"device": "cpu"},
    )
    assert state["cuda_rng_state_all"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_rng_states_are_saved_for_gpu_training(tmp_path):
    from rebuild_gpt.config import GPTConfig
    from rebuild_gpt.gpt import GPTLanguageModel
    from rebuild_gpt.tokenizer import CharTokenizer

    corpus = tmp_path / "cuda.txt"
    corpus.write_text("ababababab", encoding="utf-8")
    model = GPTLanguageModel(
        GPTConfig(2, 3, 8, 2, 1, dropout=0.2, attention_impl="fused")
    ).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "cuda.pt"
    save_text_training_state(
        path, model=model, tokenizer=CharTokenizer("ab"),
        optimizer=optimizer, corpus_path=corpus,
        controls={"device": "cuda"}, step=0,
        batch_generator=torch.Generator().manual_seed(7),
        history=[], validation_before=1.0, last_loss=1.0, last_learning_rate=0.01,
    )
    state = load_text_training_state(
        path, model=model, tokenizer=CharTokenizer("ab"),
        corpus_path=corpus, controls={"device": "cuda"},
    )
    assert isinstance(state["cuda_rng_state_all"], list)
    assert len(state["cuda_rng_state_all"]) == torch.cuda.device_count()


def test_fp16_scaler_state_is_persisted_without_accelerator(tmp_path):
    """A fake scaler checks serialization without pretending to execute CUDA."""
    corpus = tmp_path / "text.txt"
    corpus.write_text("ababab", encoding="utf-8")
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters())

    class FakeScaler:
        def state_dict(self):
            return {"scale": 128.0, "growth_tracker": 3}

    path = tmp_path / "state.pt"
    save_text_training_state(
        path, model=model, tokenizer=ToyTokenizer(), optimizer=optimizer,
        corpus_path=corpus, controls={"precision": "fp16"}, step=2,
        batch_generator=torch.Generator().manual_seed(3),
        history=[], validation_before=0.9, last_loss=0.8,
        last_learning_rate=0.01, scaler=FakeScaler(),
    )
    loaded = load_text_training_state(
        path, model=model, tokenizer=ToyTokenizer(),
        corpus_path=corpus, controls={"precision": "fp16"},
    )
    assert loaded["grad_scaler_state"] == {"scale": 128.0, "growth_tracker": 3}


def test_no_scaler_preserves_legacy_cpu_resume_compatibility(tmp_path):
    corpus = tmp_path / "text.txt"
    corpus.write_text("ababab", encoding="utf-8")
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "state.pt"
    save_text_training_state(
        path, model=model, tokenizer=ToyTokenizer(), optimizer=optimizer,
        corpus_path=corpus, controls={}, step=1,
        batch_generator=torch.Generator().manual_seed(4),
        history=[], validation_before=0.9, last_loss=0.8,
        last_learning_rate=0.01,
    )
    saved = load_text_training_state(
        path, model=model, tokenizer=ToyTokenizer(),
        corpus_path=corpus, controls={},
    )
    assert saved["grad_scaler_state"] is None
