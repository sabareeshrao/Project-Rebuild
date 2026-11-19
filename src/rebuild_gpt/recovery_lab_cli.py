"""Inspect a completed GPT recovery or run an isolated corruption drill."""

import argparse
from pathlib import Path

from rebuild_gpt.recovery_audit_verification import verify_recovery_audit
from rebuild_gpt.recovery_drill_runner import run_recovery_drill


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "drill"), default="check")
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--source-dir", default="out/multistage_resume")
    parser.add_argument("--drill-dir", default="out/recovery_drill")
    parser.add_argument("--damage-step", type=int, default=4)
    parser.add_argument("--fault-mode", choices=("append", "truncate", "remove"),
                        default="append", help="Damage a COPY, never the original")
    args = parser.parse_args(argv)
    try:
        if args.mode == "check":
            report = verify_recovery_audit(
                Path(args.source_dir) / "recovery_audit.json", args.corpus,
            )
            print("Independent recovery verification:", report["status"])
            print("Restart step:", report["restart_step"])
            print("Final step:", report["target_step"])
        else:
            kwargs = {"damage_step": args.damage_step}
            # Preserve the original call signature for the default drill.
            if args.fault_mode != "append":
                kwargs["damage_mode"] = args.fault_mode
            report = run_recovery_drill(
                args.source_dir, args.corpus, args.drill_dir, **kwargs,
            )
            print("Disposable recovery drill:", report["status"])
            print("Damaged intermediate step:", report["damage_step"])
            print("Fallback restart step:", report["fallback_step"])
            print("Independent comparison:", report["independent_check"]["status"])
            print("Original artifacts remain in:", args.source_dir)
    except (ValueError, FileNotFoundError, FileExistsError, AssertionError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
