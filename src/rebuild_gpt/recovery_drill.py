"""Prepare a disposable checkpoint-corruption drill, never edit the original."""

from pathlib import Path
from shutil import copyfile

from rebuild_gpt.resume_artifacts import (
    stage_checkpoint_path, verify_multistage_artifacts,
)
from rebuild_gpt.resume_manifest import load_multistage_manifest


def prepare_recovery_drill(
    source_dir: str | Path,
    corpus_path: str | Path,
    drill_dir: str | Path,
    *,
    damage_step: int,
) -> dict:
    """Copy verified artifacts, then corrupt ONE intermediate copy on purpose.

    Destination must not exist; the source and its original checkpoint
    files are read-only throughout this function.
    """
    source = Path(source_dir)
    destination = Path(drill_dir)
    if destination.exists():
        raise FileExistsError(f"Drill destination already exists: {destination}")
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Drill destination must not be inside the original run")
    manifest = source / "multistage_audit.json"
    report = load_multistage_manifest(manifest)
    if type(damage_step) is not int or damage_step not in report["settings"]["stops"]:
        raise ValueError("damage_step must be a recorded intermediate stop")
    # Verify the intact originals before creating ANY drill files.
    verify_multistage_artifacts(manifest, corpus_path)
    total = report["settings"]["total_steps"]
    destination.mkdir(parents=True, exist_ok=False)
    copyfile(manifest, destination / "multistage_audit.json")
    copyfile(source / "uninterrupted.pt", destination / "uninterrupted.pt")
    for entry in report["stages"]:
        name = stage_checkpoint_path(source, entry["step"], total).name
        copyfile(source / name, destination / name)

    # It is important that the copied experiment was intact beforehand.
    verify_multistage_artifacts(
        destination / "multistage_audit.json", corpus_path,
    )
    damaged = stage_checkpoint_path(destination, damage_step, total)
    with damaged.open("ab") as file:
        file.write(b"\nINTENTIONAL CORRUPTION FOR EDUCATIONAL TEST\n")
    return {
        "status": "prepared",
        "original_directory": str(source),
        "drill_directory": str(destination),
        "damaged_checkpoint": str(damaged),
        "damage_step": damage_step,
    }
