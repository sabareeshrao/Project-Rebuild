# Lesson 277: four independent checks for a saved GPT

A small loss by itself is weak evidence of correctness. Our final
conformance suite checks four different properties of a saved model:

```text
 saved GPT checkpoint + held-out corpus
       |
       +-> causal: future token cannot leak backward
       +-> isolation: other examples cannot alter logits
       +-> API: train/full and inference/last forward contracts
       +-> gradients: double-precision numerical slope vs autograd
       |
       +-> passing SHA-256-identified JSON audit
```

The test data comes from a reproducible held-out corpus window.
Gradient checking uses a float64 **copy** of the loaded GPT, so the
checkpoint weights are not modified. The report captures input hashes
and the runtime versions for further inspection.

Run:

```bash
python examples/120_capstone_cli.py --steps 4
python examples/170_model_conformance.py
```

The suite does not download or execute external nanoGPT code, and
does not substitute for true upstream parity experiments.

**Experiment:** Use the intentionally incorrect models in lessons
273–275. Which specific invariant rejects each error?

**Checkpoint:** What can these four checks establish, and what
would still be needed to claim model-quality equivalence?
