"""Lesson 256: inspect intermediate restart artifacts and their identity."""

from rebuild_gpt.resume_multistage import compare_multistage_resume
from rebuild_gpt.resume_stage_verification import verify_multistage_checkpoints


def main() -> None:
    corpus = "data/tiny_story.txt"
    run = compare_multistage_resume(
        corpus, "out/multistage_resume",
        total_steps=6, stops=(2, 4),
    )
    details = verify_multistage_checkpoints(run, corpus)
    for stage in details["stages"]:
        print(stage)
    print("final matches:", details["final_comparison"]["identical"])


if __name__ == "__main__":
    main()
