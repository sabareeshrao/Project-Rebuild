# Lesson 242: an auditable multi-step optimizer comparison

This capstone milestone checks the original nanoGPT optimizer groups
and persistent AdamW moments. Both models receive identical initial
weights and token pairs; the optimizer states persist between steps.

```text
capstone weights + held-out batch + reviewed nanoGPT model.py
                         |
                verify reference SHA-256
                         |
             compare decay group membership
                         |
           step 1 -> gradients, updated weights
           step 2 -> gradients, updated weights
           step 3 -> gradients, updated weights
                         |
                passing JSON audit report
```

```bash
python examples/120_capstone_cli.py --steps 4
python examples/135_grouped_training_audit.py \
  --upstream-model /trusted/nanoGPT/model.py \
  --upstream-sha256 <verified-source-digest> --steps 3
```

No upstream code is downloaded automatically. Importing the specified
Python file executes it, so review and pin it before running.

The study uses a repeated validation batch solely for controlled
comparison. It does not test the original training loop's data loader,
learning rate schedule, training-mode dropout, distributed execution
or large-model optimization trajectories.

**Experiment:** Repeat with 1, then 3 steps. Explain why AdamW's
moment state makes these distinct experiments.

**Checkpoint:** What would we still need to reproduce to claim
end-to-end training parity with original nanoGPT?
