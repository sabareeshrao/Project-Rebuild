"""Strictly parse benchmarking contexts to keep experiments reproducible."""


def parse_context_lengths(spec: str) -> tuple[int, ...]:
    """Accept comma-separated, unique, positive sequence lengths in order."""
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("context lengths must be a nonempty comma-separated list")
    pieces = spec.split(",")
    try:
        lengths = tuple(int(piece.strip()) for piece in pieces)
    except ValueError as exc:
        raise ValueError("context lengths must be comma-separated integers") from exc
    if any(size < 1 for size in lengths) or len(set(lengths)) != len(lengths):
        raise ValueError("context lengths must be unique positive integers")
    return lengths
