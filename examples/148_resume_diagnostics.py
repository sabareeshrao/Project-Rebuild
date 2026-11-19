"""Lesson 255: locate the first difference in two saved training states."""

from rebuild_gpt.resume_diagnostics import diagnose_resume_difference


def main() -> None:
    result = diagnose_resume_difference(
        "out/multistage_resume/uninterrupted.pt",
        "out/multistage_resume/multistage_final.pt",
    )
    print(result)


if __name__ == "__main__":
    main()
