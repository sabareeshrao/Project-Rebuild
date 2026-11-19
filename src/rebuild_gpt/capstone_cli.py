"""One-command CPU training, artifact generation, and independent replay."""

import argparse

from rebuild_gpt.capstone import save_capstone_run, train_capstone, verify_capstone_run
from rebuild_gpt.capstone_spec import CapstoneSpec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--output-dir", default="out/capstone")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prompt", default="a")
    parser.add_argument("--new-tokens", type=int, default=12)
    parser.add_argument("--verify-only", action="store_true",
                        help="Check existing files and replay sampling; no training")
    args = parser.parse_args(argv)

    try:
        if args.verify_only:
            report = verify_capstone_run(args.corpus, args.output_dir)
            print("Verified saved checkpoint, corpus and seeded sample.")
        else:
            spec = CapstoneSpec(
                steps=args.steps, seed=args.seed,
                prompt=args.prompt, new_tokens=args.new_tokens,
            )
            model, tokenizer, metrics = train_capstone(args.corpus, spec)
            report = save_capstone_run(
                model, tokenizer, metrics, args.corpus, args.output_dir, spec,
            )
            verify_capstone_run(args.corpus, args.output_dir)
            print("Trained, saved, and verified the tiny-GPT capstone.")
            print("validation before:", round(metrics["validation_before"], 4))
            print("validation after: ", round(metrics["validation_after"], 4))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    print("sample:", repr(report["generated_text"]))
    print("artifacts:", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
