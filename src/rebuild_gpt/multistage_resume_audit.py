"""Run and audit exact multi-interruption CPU GPT training equivalence."""

from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from tempfile import NamedTemporaryFile

import torch

from rebuild_gpt.resume_multistage import compare_multistage_resume
from rebuild_gpt.resume_stage_verification import verify_multistage_checkpoints


def audit_multistage_resume(
    corpus_path: str | Path,
    output_dir: str | Path,
    *,
    total_steps: int = 6,
    stops: tuple[int, ...] = (2, 4),
    seed: int = 13,
) -> dict:
    """Write a successful audit only after all saved states compare exactly.

    Removes an old report before attempting a new comparison so a failed
    rerun cannot leave a stale 'passed' audit visible to the learner.
    """
    output = Path(output_dir)
    report_path = output / "multistage_audit.json"
    report_path.unlink(missing_ok=True)
    run = compare_multistage_resume(
        corpus_path, output,
        total_steps=total_steps, stops=stops, seed=seed,
    )
    if run["status"] != "identical":
        raise AssertionError("Multi-stage run did not match uninterrupted training")
    inspected = verify_multistage_checkpoints(run, corpus_path)
    if not inspected["final_comparison"]["identical"]:
        raise AssertionError("Final checkpoint state comparison failed")

    full_hash = sha256(Path(run["uninterrupted_checkpoint"]).read_bytes()).hexdigest()
    report = {
        "format_version": 1,
        "status": "passed",
        "scope": "exact CPU fp32 grouped AdamW multi-interruption resume parity",
        "settings": {
            "total_steps": total_steps, "stops": list(stops), "seed": seed,
        },
        "sha256": {
            "corpus": inspected["corpus_sha256"],
            "uninterrupted_checkpoint": full_hash,
        },
        "stages": inspected["stages"],
        "checkpoint_comparison": inspected["final_comparison"],
        "validation_metrics": run["validation_metrics"],
        "evaluation_records": run["evaluation_records"],
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    # Atomic replacement avoids leaving a partially written success report.
    with NamedTemporaryFile(
        dir=output, prefix=".multistage_audit.", suffix=".tmp",
        mode="w", encoding="utf-8", delete=False,
    ) as temp:
        temp_path = Path(temp.name)
        try:
            json.dump(report, temp, indent=2, sort_keys=True)
            temp.write("\n")
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp_path, report_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return report
