"""Lesson 293: verify the full training -> conformance -> reproduction chain."""

from rebuild_gpt.release_evidence import verify_release_evidence


if __name__ == "__main__":
    print(verify_release_evidence("data/tiny_story.txt", "out/release_review"))
