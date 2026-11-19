"""Lesson 291: independently reproduce the trained GPT's eight checks."""

from rebuild_gpt.review_reproduction import reproduce_review_evidence


def main() -> None:
    print(reproduce_review_evidence(
        "data/tiny_story.txt", "out/release_review",
    ))


if __name__ == "__main__":
    main()
