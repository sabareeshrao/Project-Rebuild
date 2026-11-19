"""Lesson 266: restore a tiny GPT after corrupting a disposable checkpoint."""

from rebuild_gpt.recovery_drill_runner import run_recovery_drill


def main() -> None:
    report = run_recovery_drill(
        "out/multistage_resume", "data/tiny_story.txt",
        "out/recovery_drill", damage_step=4,
    )
    print("drill status:", report["status"])
    print("damaged step:", report["damage_step"])
    print("used fallback step:", report["fallback_step"])
    print("independently verified:", report["independent_check"]["status"])


if __name__ == "__main__":
    main()
