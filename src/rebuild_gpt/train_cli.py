"""Train and evaluate a small character-level GPT with reproducible logs."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.text_training import train_char_gpt
from rebuild_gpt.eval_cache_budget import estimate_evaluation_cache_budget
from rebuild_gpt.eval_index_plan import estimate_index_plan_bytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu",
                        help="Where model computation runs (default: cpu)")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32",
                        help="Computation dtype; fp16 needs CUDA and uses GradScaler")
    parser.add_argument("--attention-impl", choices=("fused", "sdpa"), default="fused",
                        help="Manual fused attention or PyTorch SDPA (default: fused)")
    parser.add_argument("--sampling-backend", choices=("indexed", "cached"), default="indexed",
                        help="Rebuild offsets per batch or reuse an unfold window view")
    parser.add_argument("--activation-checkpointing", action="store_true",
                        help="Recompute every transformer block during backward")
    parser.add_argument("--checkpoint-every-n", type=int, default=None,
                        help="Recompute every Nth transformer block (alternative to --activation-checkpointing)")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=16)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--microbatch-size", type=int, default=None,
                        help="Split one logical --batch-size batch into chunks; incompatible with accumulation steps >1")
    parser.add_argument("--pin-memory", action="store_true",
                        help="Stage logical CPU batches in pinned memory (CUDA microbatch mode only)")
    parser.add_argument("--non-blocking-transfer", action="store_true",
                        help="Use pinned CPU-to-CUDA copies on the current stream (requires --pin-memory)")
    parser.add_argument("--grouped-adamw", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--validation-mode", choices=("sampled", "full"), default="sampled")
    parser.add_argument("--eval-batch-backend", choices=("regenerate", "cached", "indices"),
                        default="regenerate",
                        help="Regenerate seeded batches, cache token windows, or store start indices")
    parser.add_argument("--eval-cache-max-mib", type=float, default=None,
                        help="Optional conservative CPU byte budget (MiB) for cached sampled evaluation")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-steps", type=int, default=None)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--output", default="out/char_gpt.pt", help="Final (or restored) inference model")
    parser.add_argument("--best-output", default="out/best_char_gpt.pt", help="Lowest validation-loss model")
    parser.add_argument("--history-file", default="out/training_history.json")
    parser.add_argument("--restore-best", action="store_true", help="Return best weights instead of final weights")
    parser.add_argument("--training-state", default=None,
                        help="Save optimizer, RNG, vocabulary and training progress")
    parser.add_argument("--resume-from", default=None,
                        help="Continue from a trusted training-state checkpoint; --steps is the total target")
    args = parser.parse_args(argv)
    if args.activation_checkpointing and args.checkpoint_every_n is not None:
        parser.error("--activation-checkpointing and --checkpoint-every-n are mutually exclusive")
    if args.microbatch_size is not None and args.gradient_accumulation_steps != 1:
        parser.error("--microbatch-size and --gradient-accumulation-steps > 1 cannot be combined")
    if (args.pin_memory or args.non_blocking_transfer) and args.microbatch_size is None:
        parser.error("--pin-memory and --non-blocking-transfer require --microbatch-size")
    if args.non_blocking_transfer and not args.pin_memory:
        parser.error("--non-blocking-transfer requires --pin-memory")
    if args.eval_batch_backend != "regenerate" and args.validation_mode != "sampled":
        parser.error("--eval-batch-backend cached/indices requires --validation-mode sampled")
    if args.eval_cache_max_mib is not None and args.eval_batch_backend == "regenerate":
        parser.error("--eval-cache-max-mib requires --eval-batch-backend cached or indices")

    paths = [Path(args.output).resolve(), Path(args.best_output).resolve(),
             Path(args.history_file).resolve()]
    if args.training_state is not None:
        paths.append(Path(args.training_state).resolve())
    if len(set(paths)) != len(paths):
        parser.error("output, best, history, and training-state paths must be distinct")
    if args.resume_from is not None and Path(args.resume_from).resolve() in paths[:3]:
        parser.error("--resume-from cannot be an inference or history output path")

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
            eval_batch_backend=args.eval_batch_backend,
            eval_cache_max_mib=args.eval_cache_max_mib,
            grouped_adamw=args.grouped_adamw,
            weight_decay=args.weight_decay,
            adam_betas=(args.adam_beta1, args.adam_beta2),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            microbatch_size=args.microbatch_size,
            pin_memory=args.pin_memory,
            non_blocking_transfer=args.non_blocking_transfer,
            resume_from=args.resume_from,
            training_state_path=args.training_state,
            device=args.device,
            precision=args.precision,
            attention_impl=args.attention_impl,
            sampling_backend=args.sampling_backend,
            activation_checkpointing=args.activation_checkpointing,
            checkpoint_every_n=args.checkpoint_every_n,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    save_text_checkpoint(model, tokenizer, args.output)
    report = {
        "corpus": str(args.corpus),
        "seed": args.seed,
        "requested_device": args.device,
        "requested_precision": args.precision,
        "model_config": asdict(model.config),
        "restored_best": args.restore_best,
        "target_steps": args.steps,
        "resumed_from": args.resume_from,
        "training_state": args.training_state,
        "training_controls": {
            "precision": args.precision,
            "activation_checkpointing": args.activation_checkpointing,
            "checkpoint_every_n": args.checkpoint_every_n,
            "grad_clip": args.grad_clip,
            "warmup_steps": args.warmup_steps,
            "lr_decay_steps": args.lr_decay_steps,
            "min_lr": args.min_lr,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "microbatch_size": args.microbatch_size,
            "pin_memory": args.pin_memory,
            "non_blocking_transfer": args.non_blocking_transfer,
            "sampling_backend": args.sampling_backend,
            "eval_batch_backend": args.eval_batch_backend,
            "eval_cache_max_mib": args.eval_cache_max_mib,
            "eval_cache_estimated_logical_bytes": (
                estimate_evaluation_cache_budget(
                    batch_size=args.batch_size, block_size=args.block_size,
                    eval_batches=args.eval_batches,
                ).total_bytes
                if args.eval_batch_backend == "cached"
                else (
                    estimate_index_plan_bytes(
                        batch_size=args.batch_size,
                        eval_batches=args.eval_batches,
                    )
                    if args.eval_batch_backend == "indices" else None
                )
            ),
            "grouped_adamw": args.grouped_adamw,
            "weight_decay": args.weight_decay if args.grouped_adamw else None,
            "adam_betas": [args.adam_beta1, args.adam_beta2] if args.grouped_adamw else None,
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
    if args.training_state is not None:
        print("resumable training state:", args.training_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
