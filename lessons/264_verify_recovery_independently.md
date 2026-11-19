# Lesson 264: independently verify the recovered training state

A recovery command may claim success, but an independent reader
should not need to rerun the optimizer to check it.

We verify:
- The original audit file and corpus bytes match the recorded hashes.
- The restart point was an audited interruption with matching bytes.
- The recovered checkpoint matches its own recorded SHA-256.
- The original uninterrupted checkpoint matches its audited digest.
- Model weights, optimizer groups/moments, RNG and history still compare
  exactly with the uninterrupted reference.

This check **loads trusted PyTorch tensor checkpoints** but does
not perform any training. It rejects changed or substituted artifacts.

Run:
`python examples/155_resume_recovery_cli.py --mode recover`
then `python examples/157_verify_recovered_state.py`.

**Experiment:** In a copy of the run directory, modify
`recovered.pt`. Which SHA-256 check detects the change?

**Checkpoint:** How does rechecking the file bytes and state differ
from simply checking a JSON field called `status`?
