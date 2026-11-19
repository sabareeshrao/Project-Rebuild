"""Lesson 269: compare a truncated checkpoint COPY with intact originals."""

from rebuild_gpt.recovery_drill import prepare_recovery_drill
from rebuild_gpt.resume_health import inspect_recovery_health


def main() -> None:
    report = prepare_recovery_drill(
        "out/multistage_resume", "data/tiny_story.txt",
        "out/truncated_checkpoint_drill", damage_step=4, damage_mode="truncate",
    )
    print("fault injected:", report["damage_mode"])
    health = inspect_recovery_health(
        "out/truncated_checkpoint_drill/multistage_audit.json",
        "data/tiny_story.txt",
    )
    print("readiness:", health["status"])
    print("usable fallback:", health["recommended_restart_step"])


if __name__ == "__main__":
    main()
