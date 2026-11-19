"""Lesson 250: run a tiny GPT both straight through and with a restart."""

from rebuild_gpt.resume_equivalence import run_resume_equivalence


def main() -> None:
    report = run_resume_equivalence(
        "data/tiny_story.txt", "out/resume_lab",
        first_steps=2, total_steps=4, seed=13,
    )
    print("comparison:", report["status"])
    print("checkpoint:", report["checkpoint_comparison"])
    print("validation:", report["final_metrics"]["validation_after"])
    print("steps:", report["first_steps"], "then", report["total_steps"])


if __name__ == "__main__":
    main()
