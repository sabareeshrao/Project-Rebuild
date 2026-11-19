"""Lesson 290: ensure training and conformance audited the same model."""

from rebuild_gpt.review_evidence_verification import verify_review_evidence


def main() -> None:
    result = verify_review_evidence(
        "data/tiny_story.txt", "out/release_review",
    )
    print(result)


if __name__ == "__main__":
    main()
