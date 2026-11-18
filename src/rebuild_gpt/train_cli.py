"""Train and evaluate a small character-level GPT with reproducible logs."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.text_training import train_char_gpt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=16)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--validation-mode", choices=("sampled", "full"), default="sampled")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-steps", type=int, default=None)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--output", default="out/char_gpt.pt", help="Final (or restored) inference model")
    parser.add_argument("--best-output", default="out/best_char_gpt.pt", help="Lowest validation-loss model")
    parser.add_argument("--history-file", default="out/training_history.json")
    parser.add_argument("--restore-best", action="store_true", help="Return best weights instead of final weights")
    args = parser.parse_args(argv)

    paths = [Path(args.output).resolve(), Path(args.best_output).resolve(),
             Path(args.history_file).resolve()]
    if len(set(paths)) != len(paths):
        parser.error("--output, --best-output, and --history-file must be distinct")

    history: list[dict[str, float | int]] = []
    try:
        model, tokenizer, metrics = train_char_gpt(
            args.corpus,
            steps=args.steps, block_size=args.block_size,
            n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            seed=args.seed, eval_interval=args.eval_every,
            eval_batches=args.eval_batches, history=history,
            restore_best=args.restore_best,
            best_checkpoint_path=args.best_output,
            max_grad_norm=args.grad_clip,
            warmup_steps=args.warmup_steps,
            lr_decay_steps=args.lr_decay_steps,
            min_lr=args.min_lr,
            validation_mode=args.validation_mode,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    save_text_checkpoint(model, tokenizer, args.output)
    report = {
        "corpus": str(args.corpus),
        "seed": args.seed,
        "model_config": asdict(model.config),
        "restored_best": args.restore_best,
        "training_controls": {
            "grad_clip": args.grad_clip,
            "warmup_steps": args.warmup_steps,
            "lr_decay_steps": args.lr_decay_steps,
            "min_lr": args.min_lr,
        },
        "validation_mode": args.validation_mode,
        "metrics": metrics,
        "evaluations": history,
    }
    output = Path(args.history_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for record in history:
        print(
            f"step={record['step']:>4}  train={record['train_loss']:.4f}"
            f"  validation={record['validation_loss']:.4f}"
        )
    print(f"best validation at step {int(metrics['best_step'])}")
    print("model checkpoint:", args.output)
    print("best checkpoint:", args.best_output)
    print("history:", args.history_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
