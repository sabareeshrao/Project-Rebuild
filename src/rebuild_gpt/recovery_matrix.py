"""Run multiple isolated GPT checkpoint failure/recovery experiments."""

from pathlib import Path

from rebuild_gpt.recovery_drill_runner import run_recovery_drill
from rebuild_gpt.resume_artifacts import verify_multistage_artifacts
from rebuild_gpt.resume_manifest import load_multistage_manifest


FAULT_MODES = ("append", "truncate", "remove")


def run_recovery_matrix(
    source_dir: str | Path,
    corpus_path: str | Path,
    output_dir: str | Path,
    *,
    modes: tuple[str, ...] = FAULT_MODES,
    damage_step: int | None = None,
) -> dict:
    """Create one disposable directory per failure scenario.

    The original audited run is read-only. Each scenario must recover
    exactly and pass an independent replay verification.
    """
    source, output = Path(source_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Matrix output already exists: {output}")
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Matrix output must not be inside the source run")
    if (not isinstance(modes, (list, tuple)) or not modes
            or any(mode not in FAULT_MODES for mode in modes)
            or len(set(modes)) != len(modes)):
        raise ValueError("modes must contain distinct supported fault modes")
    manifest = source / "multistage_audit.json"
    audit = load_multistage_manifest(manifest)
    stops = audit["settings"]["stops"]
    if len(stops) < 2:
        raise ValueError("A recovery matrix requires at least two interruption points")
    if damage_step is None:
        damage_step = stops[-1]
    if type(damage_step) is not int or damage_step != stops[-1]:
        raise ValueError("damage_step must be the latest intermediate stop")
    verify_multistage_artifacts(manifest, corpus_path)

    output.mkdir(parents=True, exist_ok=False)
    trials = []
    for mode in modes:
        trial_dir = output / mode
        result = run_recovery_drill(
            source, corpus_path, trial_dir,
            damage_step=damage_step, damage_mode=mode,
        )
        if (result.get("status") != "passed"
                or result.get("independent_check", {}).get("status") != "verified"):
            raise AssertionError(f"Recovery was not independently verified: {mode}")
        if result["fallback_step"] >= damage_step:
            raise AssertionError(f"Recovery did not select an earlier checkpoint: {mode}")
        trials.append({
            "mode": mode,
            "damage_step": damage_step,
            "fallback_step": result["fallback_step"],
            "verification": "verified",
            "directory": str(trial_dir),
        })
    return {
        "status": "passed",
        "source_directory": str(source),
        "output_directory": str(output),
        "trials": trials,
    }
