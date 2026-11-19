"""Lesson 261: resume from a verified stage and check final training state."""

from rebuild_gpt.resume_replay import replay_verified_restart


def main() -> None:
    report = replay_verified_restart(
        "out/multistage_resume/multistage_audit.json",
        "data/tiny_story.txt",
    )
    print("restarted from:", report["restart_step"])
    print("target step:", report["target_step"])
    print("result:", report["status"])
    print("checkpoint:", report["recovered_checkpoint"])


if __name__ == "__main__":
    main()
