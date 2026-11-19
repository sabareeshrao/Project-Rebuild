"""Lesson 289: train a real tiny GPT and validate eight behaviors."""

from rebuild_gpt.capstone_evidence import run_review_capstone
from rebuild_gpt.capstone_spec import CapstoneSpec


def main() -> None:
    result = run_review_capstone(
        "data/tiny_story.txt", "out/release_review",
        spec=CapstoneSpec(steps=4),
    )
    print("Tiny-GPT training and conformance:", result["status"])
    print("Model checkpoint:", result["checkpoint"])
    print("Validated checks:", result["checks_passed"])
    print("Validation metrics:", result["validation_metrics"])


if __name__ == "__main__":
    main()
