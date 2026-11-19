# Lesson 255: locate the source of training drift

A resume test that says "mismatch" is difficult to debug.
Our read-only diagnostic walks two saved checkpoint structures and
returns the name of the FIRST field that differs.

For example:

    checkpoint.optimizer.state.0.exp_avg
    checkpoint.batch_generator_state
    checkpoint.model.blocks.0.attention.weight

This helps distinguish unequal model weights from unequal optimizer
moments or batch sampling. It does not mutate model or optimizer state.

Run the multi-stage experiment first:

    python examples/147_multistage_resume.py
    python examples/148_resume_diagnostics.py

Experiment: Modify only one moment tensor in a separate copy and
run the diagnostic. Which path does it report?

Checkpoint: Why is a field-level difference more actionable than
a single boolean "different" result?
