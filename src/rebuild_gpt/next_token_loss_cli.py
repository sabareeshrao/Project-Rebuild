"""Audit saved GPT logit invariance, perplexity and token-weighted loss."""

import argparse

from rebuild_gpt.next_token_loss_audit import run_next_token_loss_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="out/release_review/capstone/model.pt")
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", default="out/release_review/next_token_loss.json")
    args = parser.parse_args(argv)
    try:
        report = run_next_token_loss_audit(
            args.checkpoint, args.corpus, seed=args.seed,
            batch_size=args.batch_size, output_path=args.output,
        )
    except (ValueError, AssertionError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print("Next-token loss audit:", report["status"])
    print("Supervised tokens:", report["checks"]["uniform_perplexity"]["supervised_positions"])
    print("Perplexity:", report["checks"]["uniform_perplexity"]["perplexity"])
    print("Masked microbatches:", report["checks"]["masked_microbatch"]["microbatch_supervised_counts"])
    print("Audit:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
