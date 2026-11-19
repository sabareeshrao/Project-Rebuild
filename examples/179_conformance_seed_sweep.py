"""Lesson 286: check model correctness across two independent random seeds."""

from rebuild_gpt.conformance_seed_sweep import run_conformance_seed_sweep


def main() -> None:
    report = run_conformance_seed_sweep(
        "out/capstone/model.pt",
        "data/tiny_story.txt",
        seeds=(11, 29),
        output_path="out/conformance/seed_sweep.json",
    )
    print("Sweep:", report["status"])
    print("Seeds checked:", report["seeds"])


if __name__ == "__main__":
    main()
