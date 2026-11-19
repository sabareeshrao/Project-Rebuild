"""Compare two inference checkpoint files on the same held-out text.

The saved inference bundles include a character vocabulary and GPT
weights; this deliberately does not load optimizer/resume checkpoints.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from rebuild_gpt.data import split_train_val
from rebuild_gpt.devices import resolve_device
from rebuild_gpt.paired_batch_evaluation import PairedBatchValidation, compare_paired_batches
from rebuild_gpt.paired_diagnostics import PairedDiagnostics, summarize_paired_deltas
from rebuild_gpt.text_checkpoint import load_text_checkpoint


@dataclass(frozen=True)
class SavedCheckpointComparison:
    baseline_path: str
    candidate_path: str
    corpus_path: str
    vocabulary_size: int
    validation_token_count: int
    baseline_config: dict
    candidate_config: dict
    paired: PairedBatchValidation
    directions: PairedDiagnostics
    sampling_policy: str
    sampling_regions: int | None


def compare_saved_checkpoints(
    baseline_path: str | Path,
    candidate_path: str | Path,
    corpus_path: str | Path,
    *,
    seeds: tuple[int, ...],
    batch_size: int,
    block_size: int,
    eval_batches: int = 4,
    device: str = "cpu",
    sampling_policy: str = "replacement",
    sampling_regions: int = 4,
) -> SavedCheckpointComparison:
    """Load trusted self-contained inference bundles and compare held-out loss.

    Require *identical character-to-ID mappings* rather than only
    matching vocabulary sizes. Loading torch modules should not
    disturb the surrounding global CPU training RNG.
    """
    compute_device = resolve_device(device)
    with torch.random.fork_rng(devices=[]):
        baseline, baseline_vocab = load_text_checkpoint(baseline_path)
        candidate, candidate_vocab = load_text_checkpoint(candidate_path)
    if baseline_vocab.vocab != candidate_vocab.vocab:
        raise ValueError("checkpoint vocabularies must match exactly (character order)")
    content = Path(corpus_path).read_text(encoding="utf-8")
    tokens = torch.tensor(baseline_vocab.encode(content), dtype=torch.long)
    _, validation = split_train_val(tokens, val_fraction=0.2)
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if validation.numel() <= block_size:
        raise ValueError("validation split is too short for block_size")
    paired = compare_paired_batches(
        baseline.to(compute_device), candidate.to(compute_device),
        validation, seeds=seeds, batch_size=batch_size,
        block_size=block_size, eval_batches=eval_batches,
        sampling_policy=sampling_policy,
        sampling_regions=sampling_regions,
    )
    return SavedCheckpointComparison(
        baseline_path=str(baseline_path),
        candidate_path=str(candidate_path),
        corpus_path=str(corpus_path),
        vocabulary_size=baseline_vocab.vocab_size,
        validation_token_count=validation.numel(),
        baseline_config=asdict(baseline.config),
        candidate_config=asdict(candidate.config),
        paired=paired,
        directions=summarize_paired_deltas(paired.differences),
        sampling_policy=sampling_policy,
        sampling_regions=sampling_regions if sampling_policy == "balanced" else None,
    )
