"""Compact, reproducible validation plans storing start positions only.

Retains one int64 vector of sampled window starts per evaluation batch
and shares the original CPU token stream; no [B,T] inputs/targets
are cached. Gathering selected windows happens at evaluation time.
"""

from dataclasses import dataclass

import torch

from rebuild_gpt.data import gather_next_token_pairs


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
