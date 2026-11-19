"""Lesson 264: independently check a completed GPT checkpoint recovery."""

from rebuild_gpt.recovery_audit_verification import verify_recovery_audit


def main() -> None:
    result = verify_recovery_audit(
        "out/multistage_resume/recovery_audit.json",
        "data/tiny_story.txt",
    )
    print("saved recovery independently verified:", result)


if __name__ == "__main__":
    main()
