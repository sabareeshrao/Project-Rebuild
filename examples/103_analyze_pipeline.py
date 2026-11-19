"""Summarize pipeline sweep CSV results, without asserting speedups."""

import argparse
import csv
import json
from pathlib import Path

from rebuild_gpt.pipeline_analysis import summarize_pipeline_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV from pipeline benchmarks")
    parser.add_argument("--output", default=None, help="Optional JSON analysis report")
    args = parser.parse_args(argv)
    source = Path(args.input)
    if args.output and source.resolve() == Path(args.output).resolve():
        parser.error("input and output must use different paths")
    try:
        with source.open(encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
        rows = summarize_pipeline_rows(records)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    report = {
        "schema_version": 1,
        "source_csv": str(source),
        "rows": rows,
        "interpretation": (
            "Fractions use sums of stage medians, which are not additive "
            "to the median end-to-end time. Stage instrumentation inserts "
            "CUDA synchronization and cannot reveal transfer/compute overlap."
        ),
    }
    for row in rows:
        print(
            f"batch={row.get('batch_size', '?')} "
            f"depth={row.get('depth', row.get('n_layer', '?'))} "
            f"largest median stage={row['largest_median_stage']}: "
            f"{row['candidate_investigation']}"
        )
    if args.output is not None:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("JSON:", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
