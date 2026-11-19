"""Lesson 285: rerun all eight checks from a verified saved GPT audit."""

from rebuild_gpt.conformance_reproduction import reproduce_conformance_report


def main() -> None:
    print(reproduce_conformance_report(
        "out/conformance/extended_conformance.json",
        "out/capstone/model.pt",
        "data/tiny_story.txt",
    ))


if __name__ == "__main__":
    main()
