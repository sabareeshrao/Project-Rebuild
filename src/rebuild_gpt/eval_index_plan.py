"""Compact, reproducible validation plans storing start positions only.

Retains one int64 vector of sampled window starts per evaluation batch
and shares the original CPU token stream; no [B,T] inputs/targets
are cached. Gathering selected windows happens at evaluation time.
"""

from dataclasses import dataclass

import torch

from rebuild_gpt.data import gather_next_token_pairs
from rebuild_gpt.eval_cache_budget import enforce_evaluation_cache_budget


@dataclass(frozen=True)
class EvaluationIndexPlan:
    tokens: torch.Tensor
    starts: tuple[torch.Tensor, ...]
    batch_size: int
    block_size: int
    seed: int

    @property
    def index_storage_bytes(self) -> int:
        """Bytes retained in start-index tensors, excluding shared corpus."""
        return sum(start.numel() * start.element_size() for start in self.starts)

    def batches(self):
        """Regather current input/target tensors in original sampled order."""
        for starts in self.starts:
            yield gather_next_token_pairs(
                self.tokens, starts, block_size=self.block_size,
            )


def prepare_evaluation_index_plan(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    eval_batches: int = 4,
    seed: int = 0,
) -> EvaluationIndexPlan:
    """Match reference sampling RNG consumption without touching global RNG."""
    for label, value in (
        ("batch_size", batch_size),
        ("block_size", block_size),
        ("eval_batches", eval_batches),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.ndim != 1 or tokens.dtype != torch.long
    ):
        raise ValueError("tokens must be a flat torch.long token stream")
    cpu_tokens = tokens.cpu()
    if len(cpu_tokens) <= block_size:
        raise ValueError("Need at least block_size + 1 tokens")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    starts = tuple(
        torch.randint(
            0, len(cpu_tokens) - block_size,
            (batch_size,), generator=rng, device="cpu",
        )
        for _ in range(eval_batches)
    )
    return EvaluationIndexPlan(
        tokens=cpu_tokens, starts=starts, batch_size=batch_size,
        block_size=block_size, seed=seed,
    )


def estimate_index_plan_bytes(
    *, batch_size: int, eval_batches: int, split_count: int = 2,
) -> int:
    """Retained int64 start-position bytes, excluding shared corpus storage."""
    for label, value in (
        ("batch_size", batch_size), ("eval_batches", eval_batches),
        ("split_count", split_count),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    return batch_size * eval_batches * split_count * torch.empty(
        (), dtype=torch.long,
    ).element_size()


def prepare_train_val_index_plans(
    train_tokens: torch.Tensor, val_tokens: torch.Tensor,
    *,
    batch_size: int, block_size: int,
    eval_batches: int = 4, seed: int = 0,
    max_bytes: int | None = None,
) -> tuple[EvaluationIndexPlan, EvaluationIndexPlan]:
    """Preflight both split shapes and combined index memory before drawing."""
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for tokens in (train_tokens, val_tokens):
        if (
            not isinstance(tokens, torch.Tensor)
            or tokens.ndim != 1 or tokens.dtype != torch.long
        ):
            raise ValueError("both token streams must be flat torch.long vectors")
        if tokens.numel() <= block_size:
            raise ValueError("both token streams need at least block_size + 1 tokens")
    bytes_needed = estimate_index_plan_bytes(
        batch_size=batch_size, eval_batches=eval_batches,
    )
    enforce_evaluation_cache_budget(bytes_needed, max_bytes)
    return (
        prepare_evaluation_index_plan(
            train_tokens, batch_size=batch_size, block_size=block_size,
            eval_batches=eval_batches, seed=seed,
        ),
        prepare_evaluation_index_plan(
            val_tokens, batch_size=batch_size, block_size=block_size,
            eval_batches=eval_batches, seed=seed + 1,
        ),
    )
