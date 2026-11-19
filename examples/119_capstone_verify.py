"""Lesson 226: verify capstone artifacts without retraining the GPT."""

from rebuild_gpt.capstone import verify_capstone_run


def main() -> None:
    print("First run: python examples/118_capstone_bundle.py")
    report = verify_capstone_run("data/tiny_story.txt", "out/capstone")
    print("checkpoint digest verified:", report["checkpoint_sha256"])
    print("seeded generation reproduced:", repr(report["generated_text"]))


if __name__ == "__main__":
    main()
