# Lesson 181 — Compare saved checkpoints from the command line

Two independent training runs often produce self-contained inference
checkpoints. We can now compare their validation behavior without
rerunning any optimization steps.

```bash
python examples/109_compare_saved_checkpoints.py \
  --baseline out/model_before.pt \
  --candidate out/model_after.pt \
  --corpus data/tiny_story.txt \
  --batch-size 4 --context 8 --eval-batches 4 \
  --first-seed 20 --seed-count 3 \
  --csv out/paired_checkpoints.csv \
  --json out/paired_checkpoints.json
```

The CLI loads both checkpoints with strict weights-only validation,
requires *identical vocabulary identity*, and evaluates on the same
contiguous held-out corpus portion and identical sampled windows.
The CSV reports per-seed mean losses, signed changes, and counts of
batches whose losses improved, worsened or tied.

JSON also saves exact GPT configs, both file paths, runtime/device
metadata, validation token count, seed selection, and descriptive
direction statistics.

A negative candidate-minus-baseline loss is improvement only on
these sampled held-out windows. It is not evidence of statistical
significance or guaranteed performance elsewhere.

The comparison never updates model weights and refuses CSV/JSON
output paths that would overwrite source checkpoints or corpus text.

Run `python -m pytest tests/test_saved_checkpoint_comparison_cli.py -q`.

**Experiment:** Compare the same checkpoint file against itself.
What should the signed deltas be for all seeds?

**Checkpoint:** Why is saving the complete vocabulary and seed
selection essential for comparing model checkpoints weeks later?
