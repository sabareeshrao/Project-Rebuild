"""Lesson 283: check the structure of an eight-invariant GPT audit."""

from rebuild_gpt.conformance_report import load_conformance_report


def main() -> None:
    report = load_conformance_report("out/conformance/extended_conformance.json")
    print("Report status:", report["status"])
    print("Checks:", ", ".join(sorted(report["checks"])))
    print("Checkpoint digest:", report["sha256"]["checkpoint"])


if __name__ == "__main__":
    main()
