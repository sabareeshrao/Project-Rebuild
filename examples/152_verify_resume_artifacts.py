"""Lesson 259: independently hash the corpus and every saved checkpoint."""

from rebuild_gpt.resume_artifacts import verify_multistage_artifacts


def main() -> None:
    result = verify_multistage_artifacts(
        "out/multistage_resume/multistage_audit.json",
        "data/tiny_story.txt",
    )
    print("status:", result["status"])
    for stage in result["stages"]:
        print(stage["step"], stage["sha256"])


if __name__ == "__main__":
    main()
