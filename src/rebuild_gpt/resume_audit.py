"""Write a source-identifying audit of real grouped-AdamW resume parity."""

from hashlib import sha256
import json
from pathlib import Path
import platform

import torch

from rebuild_gpt.resume_equivalence import run_resume_equivalence


def _sha256_file(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def audit_resume_training(
    corpus_path: str | Path,
    output_dir: str | Path,
    *,
    first_steps: int = 2,
    total_steps: int = 4,
    seed: int = 13,
) -> dict:
    """Create a 'passed' JSON record only after exact state comparison.

    The report is removed at the start so failed reruns cannot leave an old
    success marker behind. SHA-256 hashes identify file bytes, which need
    not match even when deserialized checkpoint state is identical.
    """
    output = Path(output_dir)
    report_path = output / "resume_audit.json"
    report_path.unlink(missing_ok=True)
    result = run_resume_equivalence(
        corpus_path, output, first_steps=first_steps,
        total_steps=total_steps, seed=seed,
    )
    if result["status"] != "identical" or not result["checkpoint_comparison"]["identical"]:
        raise AssertionError("Resume checkpoints do not match")
    report = {
        "format_version": 1,
        "status": "passed",
        "scope": "CPU grouped AdamW resume equivalence with seeded batches and LR schedule",
        "settings": {
            "first_steps": first_steps,
            "total_steps": total_steps,
            "seed": seed,
        },
        "sha256": {
            "corpus": _sha256_file(corpus_path),
            "full": _sha256_file(result["full_checkpoint"]),
            "partial": _sha256_file(result["partial_checkpoint"]),
            "resumed": _sha256_file(result["resumed_checkpoint"]),
        },
        "checkpoint_comparison": result["checkpoint_comparison"],
        "final_metrics": result["final_metrics"],
        "evaluation_records": result["evaluation_records"],
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
    }
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
