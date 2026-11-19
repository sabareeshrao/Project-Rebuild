"""Summarize measured stage medians without inventing speedup promises."""

import math


STAGES = ("sampling", "transfer", "update")
SUGGESTIONS = {
    "sampling": "Investigate CPU token-window gathering and input preparation.",
    "transfer": "Measure host staging and CPU-to-device transfer choices.",
    "update": "Profile attention, backward, optimizer and microbatch compute.",
}


def summarize_pipeline_rows(rows: list[dict]) -> list[dict]:
    """Annotate each report row with the largest median pipeline stage.

    Fractions use the SUM OF STAGE MEDIANS, not median end-to-end time.
    Medians of different samples are not additive in general. No
    inference about async overlap is made.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a nonempty list")
    summaries = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("every row must be a mapping")
        measurements = {}
        for stage in STAGES:
            key = f"{stage}_median_ms"
            try:
                ms = float(row[key])
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"{key} must be a finite nonnegative number") from exc
            if not math.isfinite(ms) or ms < 0:
                raise ValueError(f"{key} must be a finite nonnegative number")
            measurements[stage] = ms
        total = sum(measurements.values())
        if total <= 0:
            raise ValueError("sum of stage medians must be positive")
        if "end_to_end_median_ms" in row:
            try:
                overall = float(row["end_to_end_median_ms"])
            except (ValueError, TypeError) as exc:
                raise ValueError("end_to_end_median_ms must be finite and positive") from exc
            if not math.isfinite(overall) or overall <= 0:
                raise ValueError("end_to_end_median_ms must be finite and positive")
        slowest = max(STAGES, key=lambda stage: measurements[stage])
        summaries.append({
            **row,
            "largest_median_stage": slowest,
            "stage_fraction_of_median_sum": {
                stage: measurements[stage] / total for stage in STAGES
            },
            "candidate_investigation": SUGGESTIONS[slowest],
        })
    return summaries
