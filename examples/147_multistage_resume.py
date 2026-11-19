"""Lesson 254: real six-step training with interruptions after 2 and 4."""

from rebuild_gpt.resume_multistage import compare_multistage_resume


def main() -> None:
    report = compare_multistage_resume(
        "data/tiny_story.txt", "out/multistage_resume",
        total_steps=6, stops=(2, 4), seed=13,
    )
    print("result:", report["status"])
    print("saved stages:", report["stage_checkpoints"])
    print("matching model, AdamW and RNG:", report["checkpoint_comparison"])


if __name__ == "__main__":
    main()
