"""Query-key building blocks for causal self-attention."""

import torch
from torch import nn

from rebuild_gpt.probability import stable_softmax


def pairwise_dot_scores(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Match each query with each key: [B,T,D] x [B,S,D] -> [B,T,S]."""
    if (
        queries.ndim != 3 or keys.ndim != 3
        or queries.numel() == 0 or keys.numel() == 0
        or not queries.is_floating_point() or not keys.is_floating_point()
        or queries.shape[0] != keys.shape[0]
        or queries.shape[-1] != keys.shape[-1]
    ):
        raise ValueError("Expected floating-point queries [B,T,D] and keys [B,S,D] with matching B,D")
    return queries @ keys.transpose(-2, -1)


class QueryKeyProjection(nn.Module):
    """Learn separate query and key vectors from each input token vector."""

    def __init__(self, n_embd: int, head_size: int) -> None:
        super().__init__()
        if n_embd < 1 or head_size < 1:
            raise ValueError("n_embd and head_size must be positive")
        self.n_embd = n_embd
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            hidden.ndim != 3 or hidden.numel() == 0
            or not hidden.is_floating_point() or hidden.size(-1) != self.n_embd
        ):
            raise ValueError("hidden must be floating-point [B,T,n_embd]")
        return self.query(hidden), self.key(hidden)


def scaled_dot_scores(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Scale pairwise dot products by inverse sqrt of query/key width."""
    scores = pairwise_dot_scores(queries, keys)
    return scores * (queries.size(-1) ** -0.5)


def causal_mask_scores(scores: torch.Tensor) -> torch.Tensor:
    """Block future keys by replacing upper-triangle logits with -inf."""
    if (
        scores.ndim != 3 or scores.numel() == 0
        or not scores.is_floating_point() or scores.size(-2) != scores.size(-1)
    ):
        raise ValueError("scores must be floating-point square [B,T,T]")
    time = scores.size(-1)
    future = torch.ones(time, time, dtype=torch.bool, device=scores.device).triu(1)
    return scores.masked_fill(future, float("-inf"))


class CausalAttentionWeights(nn.Module):
    """Produce causal query-key mixing weights, before applying values."""

    def __init__(self, n_embd: int, head_size: int) -> None:
        super().__init__()
        self.projections = QueryKeyProjection(n_embd, head_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        queries, keys = self.projections(hidden)
        scores = scaled_dot_scores(queries, keys)
        return stable_softmax(causal_mask_scores(scores))


class ValueProjection(nn.Module):
    """Learn the content vectors to be mixed by attention weights."""

    def __init__(self, n_embd: int, head_size: int) -> None:
        super().__init__()
        if n_embd < 1 or head_size < 1:
            raise ValueError("n_embd and head_size must be positive")
        self.n_embd = n_embd
        self.value = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if (
            hidden.ndim != 3 or hidden.numel() == 0
            or not hidden.is_floating_point() or hidden.size(-1) != self.n_embd
        ):
            raise ValueError("hidden must be floating-point [B,T,n_embd]")
        return self.value(hidden)
