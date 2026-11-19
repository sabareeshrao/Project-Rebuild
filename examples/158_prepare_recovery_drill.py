"""Lesson 265: intentionally corrupt a COPY of one saved GPT checkpoint."""

from rebuild_gpt.recovery_drill import prepare_recovery_drill


def main() -> None:
    result = prepare_recovery_drill(
        "out/multistage_resume", "data/tiny_story.txt",
        "out/recovery_drill", damage_step=4,
    )
    print(result)
    print("Original checkpoint files were not modified.")


if __name__ == "__main__":
    main()
