"""Lesson 258: read a saved audit's declared steps and checkpoint digests."""

from rebuild_gpt.resume_manifest import load_multistage_manifest


def main() -> None:
    path = "out/multistage_resume/multistage_audit.json"
    report = load_multistage_manifest(path)
    print("planned stops:", report["settings"]["stops"])
    print("total steps:", report["settings"]["total_steps"])
    print("checkpoint digests:")
    for stage in report["stages"]:
        print(" ", stage["step"], stage["sha256"])


if __name__ == "__main__":
    main()
