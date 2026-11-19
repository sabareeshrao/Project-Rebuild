"""Lesson 271: compare three independent audited checkpoint failures."""

from rebuild_gpt.recovery_matrix import run_recovery_matrix


def main() -> None:
    report = run_recovery_matrix(
        "out/multistage_resume", "data/tiny_story.txt",
        "out/recovery_matrix",
    )
    print("matrix result:", report["status"])
    for trial in report["trials"]:
        print(trial["mode"], "-> fallback", trial["fallback_step"])


if __name__ == "__main__":
    main()
