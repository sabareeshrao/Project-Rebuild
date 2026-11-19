"""Lesson 270: delete a copied checkpoint, then recover the same GPT state."""

from rebuild_gpt.recovery_drill_runner import run_recovery_drill


def main() -> None:
    report = run_recovery_drill(
        "out/multistage_resume", "data/tiny_story.txt",
        "out/recovery_missing_drill",
        damage_step=4, damage_mode="remove",
    )
    print("fault:", report["damage_mode"])
    print("fallback:", report["fallback_step"])
    print("recovered:", report["independent_check"]["status"])


if __name__ == "__main__":
    main()
