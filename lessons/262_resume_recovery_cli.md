# Lesson 262: an auditable recovery workflow

We now have two explicitly separate operations:

- **Verify** only hashes known-good data and checkpoints. It does not
  import a model, load a `.pt` file or execute the optimizer.
- **Recover** chooses a hash-matching intermediate checkpoint, resumes
  the existing GPT trainer, and compares the final saved model,
  AdamW state and RNG history with the original uninterrupted run.

    trusted manifest + saved files
      |--> verify: read-only SHA-256 checks
      |--> recover: select last good stage -> resume -> compare
                                                  |
                                          recovery_audit.json

Generate a trusted audit first:

    python examples/150_multistage_resume_audit.py
    python examples/155_resume_recovery_cli.py --mode verify
    python examples/155_resume_recovery_cli.py --mode recover

**Experiment:** Work on a copy of the output directory. Modify the
step-4 checkpoint. Verify mode must fail; recover mode should select
an earlier intact stage and compare the replayed training result.

**Checkpoint:** Why must recovery use both the original CPU/batch RNG
states and the optimizer moments, not just the model weights?

**Security:** The SHA-256 audit is not signed. Treat the manifest and
all loaded PyTorch checkpoints as trusted local experimental artifacts.
