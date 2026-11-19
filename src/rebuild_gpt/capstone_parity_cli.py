"""Compare saved capstone outputs to a user-reviewed original nanoGPT source.

Warning: importing --upstream-model EXECUTES that local Python file.
"""

import argparse

from rebuild_gpt.capstone_parity_audit import run_capstone_parity_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-model", required=True,
                        help="Path to a trusted, reviewed original nanoGPT model.py")
    parser.add_argument("--upstream-sha256", default=None,
                        help="Optional 64-character SHA-256 pin; checked before importing")
    parser.add_argument("--checkpoint", default="out/capstone/model.pt")
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--report", default="out/capstone/upstream_parity.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--prompt", default="a")
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        report = run_capstone_parity_audit(
            args.checkpoint,
            args.corpus,
            args.upstream_model,
            output_path=args.report,
            expected_upstream_sha256=args.upstream_sha256,
            seed=args.seed,
            batch_size=args.batch_size,
            prompt=args.prompt,
            new_tokens=args.new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    except (ValueError, FileNotFoundError, AssertionError) as exc:
        parser.error(str(exc))
    print("Forward and generation parity:", report["status"])
    print("Matched parameter tensors:",
          report["checks"]["forward_logits_and_loss"]["matched_tensors"])
    print("Upstream SHA-256:", report["sha256"]["upstream_model_py"])
    print("Audit report:", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
