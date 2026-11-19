"""Lesson 284: check the actual model and corpus bytes against an audit."""

from rebuild_gpt.conformance_integrity import verify_conformance_integrity


def main() -> None:
    report = verify_conformance_integrity(
        "out/conformance/extended_conformance.json",
        "out/capstone/model.pt",
        "data/tiny_story.txt",
    )
    print(report)


if __name__ == "__main__":
    main()
