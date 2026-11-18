# Lesson 77: make experiments repeatable and inspectable

We now have an end-to-end command that trains, periodically evaluates,
saves the **final model**, keeps the **lowest validation-loss model**,
and writes a plain JSON record of the experiment.

```text
corpus -> GPT training -> seeded evaluations at steps 0, 10, 20, ...
                         |               |
                         |               +-> best model + tokenizer .pt
                         +-> metrics + config + seed -> JSON log
```

Run the full example:

```bash
python examples/74_monitored_training.py
```

Or train once and generate in a separate command:

```bash
python -m rebuild_gpt.train_cli \
  --corpus data/tiny_story.txt --steps 60 \
  --eval-every 10 --eval-batches 4 --seed 7 \
  --output out/char_gpt.pt --best-output out/best_char_gpt.pt \
  --history-file out/training_history.json

python -m rebuild_gpt.text_cli \
  --checkpoint out/best_char_gpt.pt --prompt "a" \
  --new-tokens 80 --temperature 0.8 --top-k 5 --seed 13
```

The JSON log includes the model config, seed, train/validation
estimates at each evaluation step, and the best step. Use
`--restore-best` if the model stored in `--output` should also
contain the best checkpoint weights instead of final-step weights.

**Experiment:** Run the exact command twice and compare the
JSON logs. Then change the seed and inspect what changes.

**Checkpoint:** Why is a stable validation comparison more useful
for selecting a checkpoint than simply saving the last step?

**Limitations:** Evaluation samples fixed mini-batches rather than
exhaustively scoring the corpus. This CPU-scale trainer does not
yet include learning-rate schedules, precision controls or large
real-world language-model benchmarks.
