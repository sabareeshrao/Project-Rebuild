"""Lesson 260: select the newest intact restart point from audited files."""

from rebuild_gpt.resume_recovery import select_verified_restart


def main() -> None:
    recovery = select_verified_restart(
        "out/multistage_resume/multistage_audit.json",
        "data/tiny_story.txt",
    )
    print("checkpoint:", recovery["checkpoint"])
    print("restart from step:", recovery["step"])
    print("resume toward step:", recovery["target_step"])
    print("unusable later stages:", recovery["skipped_later_stages"])


if __name__ == "__main__":
    main()
