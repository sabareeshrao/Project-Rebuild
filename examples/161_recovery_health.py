"""Lesson 268: inspect readiness of audited GPT checkpoints."""

from rebuild_gpt.resume_health import inspect_recovery_health


def main() -> None:
    report = inspect_recovery_health(
        "out/multistage_resume/multistage_audit.json",
        "data/tiny_story.txt",
    )
    print("readiness:", report["status"])
    print("recommended restart:", report["recommended_restart_step"])
    for stage in report["stages"]:
        print(stage["step"], stage["status"])


if __name__ == "__main__":
    main()
