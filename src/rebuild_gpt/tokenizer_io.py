"""Store character vocabularies as simple, versioned UTF-8 JSON."""

import json
from pathlib import Path

from rebuild_gpt.tokenizer import CharTokenizer


TOKENIZER_FORMAT_VERSION = 1


def save_tokenizer(tokenizer: CharTokenizer, path: str | Path) -> None:
    """Persist the exact ID order instead of rebuilding from new text."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": TOKENIZER_FORMAT_VERSION, "vocab": list(tokenizer.vocab)}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_tokenizer(path: str | Path) -> CharTokenizer:
    """Validate the saved vocabulary before interpreting any token IDs."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format_version") != TOKENIZER_FORMAT_VERSION:
        raise ValueError("Unsupported tokenizer format_version")
    return CharTokenizer.from_vocab(payload.get("vocab"))
