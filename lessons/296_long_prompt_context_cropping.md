# Lesson 296 — Keep the prompt, crop only the attention context

GPT has a fixed maximum attention window. The generated **output**
can be much longer, but each next-token forward pass sees only the
most recent `block_size` token IDs.

Our test deliberately starts with seven tokens and a block size of
four:

```text
original prompt: 1 2 3 4 5 6 7
                 |-> first attention window: 4 5 6 7
sample token 8   |-> next  attention window: 5 6 7 8
return:          1 2 3 4 5 6 7 8 ...
```

An independent manual loop uses the same seed and sampling distribution
while selecting its context window explicitly. It compares all
generated IDs with the real `GPTLanguageModel.generate` result.

Run `python examples/189_generation_context_check.py`.

**Question:** Why is it wrong to discard the early prompt tokens
from the **returned text**, even though they no longer fit in attention?
