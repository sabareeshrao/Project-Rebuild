"""Lesson 249: compare the two checkpoints from the resume capstone."""

import argparse

from rebuild_gpt.resume_comparison import compare_resume_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full", nargs="?", default="out/resume_lab/full.pt")
    parser.add_argument("resumed", nargs="?", default="out/resume_lab/resumed.pt")
    args = parser.parse_args()
    print(compare_resume_checkpoints(args.full, args.resumed))


if __name__ == "__main__":
    main()
