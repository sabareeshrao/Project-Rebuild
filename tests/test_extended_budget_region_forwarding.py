"""Extended checkpoint budgets pass custom region settings only to balanced."""

from pathlib import Path


def test_extended_checkpoint_sweep_forwards_region_count_only_when_needed():
    # This precise contract guards against the former CI failure:
    # non-balanced policies reject a nondefault sampling_regions value.
    source = (Path(__file__).resolve().parents[1]
              / "examples/110_paired_budget_sweep.py").read_text(
                  encoding="utf-8",
              )
    assert 'sampling_regions=(region_count if policy == "balanced" else 4)' in source
    assert '"extended": ("replacement", "disjoint", "shifted", "balanced")' in source
