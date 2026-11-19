"""Lesson 288: verify training artifact bytes without loading the model."""

from rebuild_gpt.capstone_integrity import verify_capstone_artifacts


def main() -> None:
    print(verify_capstone_artifacts(
        "data/tiny_story.txt", "out/capstone",
    ))


if __name__ == "__main__":
    main()
