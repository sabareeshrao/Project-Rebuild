# Lesson 140 — Connect streaming to real text training

The previous microbatch CLI sampled one logical batch on CPU,
but it transferred every slice to the accelerator before processing
any of them.

Now `train_char_gpt(..., microbatch_size=M)` samples the **same CPU
logical batch**, then passes it to the new streamed training helper.
Only the current microbatch is transferred before its forward and
backward pass. The next chunk is not transferred until the previous
chunk has finished backpropagating.

```text
CPU sample B=5 → stream chunk 2 → accelerator → backward
               → stream chunk 2 → accelerator → backward
               → stream chunk 1 → accelerator → backward
                                                       ↓
                                                  one AdamW step
```

The default path, multi-sample `gradient_accumulation_steps` mode,
validation, LR schedule, and saved training controls are unchanged.
There is no new CLI flag: this is a safer implementation of the
existing `--microbatch-size` behavior.

Tests check that the helper receives the logical batch on CPU,
that streamed and unsplit models match within normal rounding,
and that a checkpointed/resumed stream reproduces an uninterrupted run.

Run `python -m pytest tests/test_streamed_text_training.py -q`.

**Experiment:** Train with B=5 and M=2 on CUDA and compare the
allocator peak against B=5 and M=5.

**Checkpoint:** Why do we preserve CPU batch-sampling RNG state
even though its tensors may later be transferred to CUDA?
