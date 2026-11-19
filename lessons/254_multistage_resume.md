# Lesson 254: prove that two training restarts are exact

Run the same six-step experiment through two execution paths:

    uninterrupted: step 1 ... step 6 -> uninterrupted.pt

    interrupted:   step 1-2 -> save
                   step 3-4 -> save
                   step 5-6 -> multistage_final.pt

The existing trainer restores model weights, AdamW moments, CPU
and batch-sampling random streams, and learning-rate controls.

Evaluation runs at every step, avoiding differences in evaluation
cadence between early-stage and uninterrupted runs.

After training, we compare the *deserialized meaningful state*
exactly: model parameters, optimizer moments, RNG streams, controls,
history and metrics. This is a deterministic CPU fp32 experiment,
not a guarantee for every PyTorch/CUDA environment.

Run: python examples/147_multistage_resume.py

Checkpoint: Why is it important to keep the same final learning-rate
schedule even when an intermediate stage ends sooner?
