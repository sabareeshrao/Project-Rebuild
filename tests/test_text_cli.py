import pytest

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.text_cli import main
from rebuild_gpt.text_generation import generate_text
from rebuild_gpt.text_training import train_char_gpt
from rebuild_gpt.tokenizer import CharTokenizer


def test_cli_loads_saved_vocabulary_and_generates_reproducibly(tmp_path, capsys):
    tok = CharTokenizer("ab.")
    model = GPTLanguageModel(GPTConfig(tok.vocab_size, 6, 8, 2, 1)).eval()
    path = tmp_path / "inference.pt"
    save_text_checkpoint(model, tok, path)
    args = [
        "--checkpoint", str(path), "--prompt", "ab", "--new-tokens", "4",
        "--top-k", "2", "--seed", "11",
    ]
    assert main(args) == 0
    first = capsys.readouterr().out.strip()
    assert main(args) == 0
    second = capsys.readouterr().out.strip()
    assert first == second
    assert first.startswith("ab")
    assert len(first) == 6


def test_cli_reports_unseen_prompt_characters(tmp_path, capsys):
    tok = CharTokenizer("ab")
    model = GPTLanguageModel(GPTConfig(tok.vocab_size, 4, 8, 2, 1))
    path = tmp_path / "model.pt"
    save_text_checkpoint(model, tok, path)
    with pytest.raises(SystemExit) as error:
        main(["--checkpoint", str(path), "--prompt", "z"])
    assert error.value.code == 2
    assert "Character" in capsys.readouterr().err


def test_cli_rejects_negative_seed(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--checkpoint", str(tmp_path / "unused.pt"), "--prompt", "a", "--seed", "-1"])
    assert error.value.code == 2
    assert "--seed" in capsys.readouterr().err


def test_training_to_saved_checkpoint_to_text_output(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("ab" * 80, encoding="utf-8")
    model, tok, metrics = train_char_gpt(
        corpus, steps=15, block_size=4, n_embd=8, n_head=2, n_layer=1,
        batch_size=8, learning_rate=0.03, seed=3,
    )
    path = tmp_path / "trained.pt"
    save_text_checkpoint(model, tok, path)
    assert metrics["validation_after"] < metrics["validation_before"]
    from rebuild_gpt.text_checkpoint import load_text_checkpoint
    reloaded, restored_tok = load_text_checkpoint(path)
    text = generate_text(reloaded, restored_tok, "ab", 8, top_k=1)
    assert text.startswith("ab")
    assert len(text) == 10
    assert set(text) <= {"a", "b"}
