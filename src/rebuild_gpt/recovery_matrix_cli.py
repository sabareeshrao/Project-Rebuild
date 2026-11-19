"""Run a SHA-256-identified matrix of independent GPT recovery failures."""

import argparse
from pathlib import Path

from rebuild_gpt.recovery_matrix_audit import audit_recovery_matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="out/multistage_resume")
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--matrix-dir", default="out/recovery_matrix")
    parser.add_argument("--modes", default="append,truncate,remove",
                        help="Distinct comma-separated append,truncate,remove modes")
    parser.add_argument("--damage-step", type=int, default=None,
                        help="Latest intermediate checkpoint step; autodetect by default")
    args = parser.parse_args(argv)
    modes = tuple(token.strip() for token in args.modes.split(","))
    try:
        report = audit_recovery_matrix(
            args.source_dir, args.corpus, args.matrix_dir,
            modes=modes, damage_step=args.damage_step,
        )
    except (ValueError, FileExistsError, FileNotFoundError, AssertionError) as exc:
        parser.error(str(exc))
    print("Recovery fault matrix:", report["status"])
    for trial in report["trials"]:
        print(trial["mode"], "fallback step:", trial["fallback_step"])
    print("Audit:", Path(args.matrix_dir) / "recovery_matrix_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
