"""Repeat all eight GPT checks across independent deterministic token draws."""

import json
from pathlib import Path

from rebuild_gpt.extended_conformance import run_extended_conformance


def run_conformance_seed_sweep(
    checkpoint_path: str | Path,
    corpus_path: str | Path,
    *,
    seeds: tuple[int, ...] = (11, 29),
    batch_size: int = 2,
    output_path: str | Path | None = None,
) -> dict:
    """Require every selected seed to pass without conflating sampled tokens.

    All rounds check the SAME immutable checkpoint and corpus. Each seed
    samples its own held-out windows and private generation stream.
    The report only appears when the complete sweep succeeds.
    """
    if (not isinstance(seeds, (list, tuple)) or len(seeds) < 2
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be at least two distinct nonnegative integers")
    if type(batch_size) is not int or batch_size < 2:
        raise ValueError("batch_size must be an integer >= 2")
    if output_path is not None:
        Path(output_path).unlink(missing_ok=True)

    rounds: list[dict] = []
    reference_hashes = None
    reference_model = None
    for seed in seeds:
        report = run_extended_conformance(
            checkpoint_path, corpus_path, seed=seed, batch_size=batch_size,
        )
        if report.get("status") != "passed" or len(report.get("checks", {})) != 8:
            raise AssertionError(f"Eight conformance checks did not pass at seed {seed}")
        if reference_hashes is None:
            reference_hashes = report["sha256"]
            reference_model = report["model"]
        elif report["sha256"] != reference_hashes or report["model"] != reference_model:
            raise AssertionError("The checkpoint or model changed during the seed sweep")
        rounds.append({
            "seed": seed,
            "checks": report["checks"],
        })
    output = {
        "format_version": 1,
        "status": "passed",
        "scope": "eight GPT conformance checks across distinct deterministic seeds",
        "seeds": list(seeds),
        "batch_size": batch_size,
        "sha256": reference_hashes,
        "model": reference_model,
        "rounds": rounds,
    }
    if output_path is not None:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    return output
