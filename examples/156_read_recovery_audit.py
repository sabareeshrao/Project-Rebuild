"""Lesson 263: inspect recovery evidence without loading any checkpoints."""

from rebuild_gpt.recovery_audit_schema import load_recovery_audit


def main() -> None:
    audit = load_recovery_audit("out/multistage_resume/recovery_audit.json")
    recovery = audit["recovery"]
    print("status:", audit["status"])
    print("restart step:", recovery["restart_step"])
    print("final step:", recovery["target_step"])
    print("recovered SHA-256:", audit["recovered_checkpoint_sha256"])


if __name__ == "__main__":
    main()
