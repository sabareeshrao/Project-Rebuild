"""Train, sample, verify and test one tiny GPT on a trusted CPU corpus."""

from pathlib import Path

from rebuild_gpt.capstone import save_capstone_run, train_capstone, verify_capstone_run
from rebuild_gpt.capstone_integrity import verify_capstone_artifacts
from rebuild_gpt.capstone_spec import CapstoneSpec
from rebuild_gpt.extended_conformance import run_extended_conformance


def run_review_capstone(
    corpus_path: str | Path,
    output_dir: str | Path,
    *,
    spec: CapstoneSpec | None = None,
    conformance_seed: int = 29,
    batch_size: int = 2,
) -> dict:
    """Produce a real training checkpoint and eight-check conformance audit.

    Use a NEW output directory to avoid overwriting earlier evidence.
    This is a small CPU learning experiment, not original GPT-2 training.
    """
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Review output directory already exists: {output}")
    if spec is None:
        spec = CapstoneSpec()
    if not isinstance(spec, CapstoneSpec):
        raise ValueError("spec must be a CapstoneSpec")
    if type(conformance_seed) is not int or conformance_seed < 0:
        raise ValueError("conformance_seed must be nonnegative")
    if type(batch_size) is not int or batch_size < 2:
        raise ValueError("batch_size must be at least two")
    if not Path(corpus_path).is_file():
        raise FileNotFoundError(f"Corpus does not exist: {corpus_path}")
    capstone_dir = output / "capstone"
    model, tokenizer, metrics = train_capstone(corpus_path, spec)
    manifest = save_capstone_run(
        model, tokenizer, metrics, corpus_path, capstone_dir, spec,
    )
    reproduced = verify_capstone_run(corpus_path, capstone_dir)
    if manifest != reproduced:
        raise AssertionError("The capstone sample replay changed the manifest")
    verified = verify_capstone_artifacts(corpus_path, capstone_dir)
    report_path = output / "extended_conformance.json"
    conformance = run_extended_conformance(
        capstone_dir / "model.pt", corpus_path,
        batch_size=batch_size, seed=conformance_seed,
        output_path=report_path,
    )
    if conformance.get("status") != "passed" or len(conformance.get("checks", {})) != 8:
        raise AssertionError("Eight model conformance checks did not pass")
    if (conformance["sha256"]["checkpoint"] != verified["checkpoint_sha256"]
            or conformance["sha256"]["corpus"] != verified["corpus_sha256"]):
        raise AssertionError("Conformance checked different model or corpus bytes")
    return {
        "status": "passed",
        "capstone_dir": str(capstone_dir),
        "checkpoint": str(capstone_dir / "model.pt"),
        "conformance_report": str(report_path),
        "capstone_integrity": verified,
        "validation_metrics": metrics,
        "checks_passed": len(conformance["checks"]),
        "conformance_seed": conformance_seed,
    }
