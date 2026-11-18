# Lesson 49: initialize weights on the right scale

Model initialization influences early gradients and stability.
A GPT-like starting point is:

- Linear and embedding weights: normal distribution with standard deviation 0.02.
- Linear biases: zeros.
- LayerNorm scale: ones; LayerNorm bias: zeros.
- Residual output projections: standard deviation `0.02 / sqrt(2 * n_layer)`.

Each transformer block has two residual branches: attention and MLP.
Scaling their output projections makes very deep residual stacks start
with smaller perturbations. This is an initialization convention, not
a guarantee of successful training.

**Important:** We must initialize a shared token/output weight only once.
The helper records Parameter identities to avoid applying two different
random initializations to one tied tensor.

Run: `python examples/49_gpt_initialization.py`.

**Experiment:** Change `n_layer` from 2 to 8. What happens to the
target initialization standard deviation for residual projections?

**Checkpoint:** Why should the output projection in a residual branch
start with a smaller scale than a typical Q/K/V matrix?

We will apply the same scheme to the fused attention implementation
introduced in the next lessons.
