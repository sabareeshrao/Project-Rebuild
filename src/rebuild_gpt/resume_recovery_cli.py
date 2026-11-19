"""Read-only integrity verification or exact GPT training recovery from audit."""

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from rebuild_gpt.resume_artifacts import sha256_file, verify_multistage_artifacts
from rebuild_gpt.resume_replay import replay_verified_restart


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="out/multistage_resume")
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--mode", choices=("verify", "recover"), default="verify",
                        help="verify: hash only; recover: run the optimizer")
    args = parser.parse_args(argv)

    directory = Path(args.run_dir)
    manifest = directory / "multistage_audit.json"
    try:
        if args.mode == "verify":
            report = verify_multistage_artifacts(manifest, args.corpus)
            print("Artifact integrity:", report["status"])
            print("Verified stages:", [x["step"] for x in report["stages"]])
        else:
            destination = directory / "recovery_audit.json"
            # A failed new recovery must not leave yesterday's passing report.
            destination.unlink(missing_ok=True)
            result = replay_verified_restart(manifest, args.corpus)
            if (result.get("status") != "identical"
                    or result.get("checkpoint_comparison", {}).get("identical") is not True):
                raise AssertionError("Recovered training did not match reference state")
            audit = {
                "format_version": 1,
                "status": "passed",
                "scope": "saved GPT checkpoint recovery to original CPU training state",
                "source_manifest_sha256": sha256_file(manifest),
                "recovered_checkpoint_sha256": sha256_file(
                    result["recovered_checkpoint"]
                ),
                "recovery": result,
            }
            directory.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory,
                prefix=".recovery_audit.", suffix=".tmp", delete=False,
            ) as temp:
                temp_path = Path(temp.name)
            try:
                temp_path.write_text(
                    json.dumps(audit, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(temp_path, destination)
            finally:
                temp_path.unlink(missing_ok=True)
            print("Recovered training state:", result["status"])
            print("Reused saved step:", result["restart_step"])
            print("Skipped damaged stages:", result["skipped_later_stages"])
            print("Audit:", destination)
    except (ValueError, FileNotFoundError, AssertionError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
