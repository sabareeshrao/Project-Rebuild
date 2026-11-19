"""Lesson 251: generate a SHA-256-identified resume equivalence audit."""

from rebuild_gpt.resume_audit import audit_resume_training


def main() -> None:
    report = audit_resume_training(
        "data/tiny_story.txt", "out/resume_lab",
        first_steps=2, total_steps=4, seed=13,
    )
    print("result:", report["status"])
    print("state:", report["checkpoint_comparison"])
    print("SHA-256 identifiers:", report["sha256"])
    print("report: out/resume_lab/resume_audit.json")


if __name__ == "__main__":
    main()
