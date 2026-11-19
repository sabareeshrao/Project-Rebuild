"""A small learning-first training entrypoint built on the existing trainer."""

from pathlib import Path

from rebuild_gpt.capstone_spec import CapstoneSpec
from rebuild_gpt.text_training import train_char_gpt


def train_capstone(corpus_path: str | Path, spec: CapstoneSpec):
    """Train a tiny real GPT; keep the existing training path authoritative."""
    return train_char_gpt(corpus_path, **spec.training_kwargs())


def _sha256_file(path: Path) -> str:
    """Digest exact bytes, including corpus newlines and checkpoint metadata."""
    from hashlib import sha256

    return sha256(path.read_bytes()).hexdigest()


def save_capstone_run(
    model,
    tokenizer,
    metrics: dict,
    corpus_path: str | Path,
    output_dir: str | Path,
    spec: CapstoneSpec,
) -> dict:
    """Save a self-contained model, deterministic text sample and audit manifest."""
    from dataclasses import asdict
    import json

    import torch

    from rebuild_gpt.text_checkpoint import load_text_checkpoint, save_text_checkpoint
    from rebuild_gpt.text_generation import generate_text

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    sample_path = output / "sample.txt"
    manifest_path = output / "run.json"

    save_text_checkpoint(model, tokenizer, checkpoint)
    restored, restored_tokenizer = load_text_checkpoint(checkpoint)
    sample = generate_text(
        restored, restored_tokenizer, spec.prompt, spec.new_tokens,
        temperature=spec.temperature, top_k=spec.top_k,
        generator=torch.Generator().manual_seed(spec.seed + 1),
    )
    sample_path.write_text(sample, encoding="utf-8")
    manifest = {
        "format_version": 1,
        "checkpoint": "model.pt",
        "settings": asdict(spec),
        "metrics": dict(metrics),
        "corpus_sha256": _sha256_file(Path(corpus_path)),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "generated_text": sample,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_capstone_run(corpus_path: str | Path, output_dir: str | Path) -> dict:
    """Verify bytes and replay sampling without any training or optimizer steps."""
    import json

    import torch

    from rebuild_gpt.text_checkpoint import load_text_checkpoint
    from rebuild_gpt.text_generation import generate_text

    output = Path(output_dir)
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or manifest.get("checkpoint") != "model.pt":
        raise ValueError("Unsupported capstone manifest format or checkpoint name")
    if _sha256_file(Path(corpus_path)) != manifest["corpus_sha256"]:
        raise ValueError("Corpus SHA-256 mismatch")
    checkpoint = output / "model.pt"
    if _sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 mismatch")
    if (output / "sample.txt").read_text(encoding="utf-8") != manifest["generated_text"]:
        raise ValueError("Saved sample does not match manifest")

    spec = CapstoneSpec(**manifest["settings"])
    restored, tokenizer = load_text_checkpoint(checkpoint)
    reproduced = generate_text(
        restored, tokenizer, spec.prompt, spec.new_tokens,
        temperature=spec.temperature, top_k=spec.top_k,
        generator=torch.Generator().manual_seed(spec.seed + 1),
    )
    if reproduced != manifest["generated_text"]:
        raise ValueError("Generated text cannot be reproduced from the saved model")
    return manifest
