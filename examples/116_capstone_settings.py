"""Lesson 223: inspect a reproducible, deliberately tiny GPT recipe."""

from dataclasses import asdict

from rebuild_gpt.capstone_spec import CapstoneSpec


def main() -> None:
    spec = CapstoneSpec()
    print("full experiment:", asdict(spec))
    print("CPU training options:", spec.training_kwargs())


if __name__ == "__main__":
    main()
