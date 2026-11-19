"""Lesson 253: draw a valid multi-interruption training plan."""

from rebuild_gpt.resume_milestones import training_milestones


def main() -> None:
    points = training_milestones(6, (2, 4))
    previous = 0
    for step in points:
        print(f"train steps {previous + 1} through {step}")
        previous = step


if __name__ == "__main__":
    main()
