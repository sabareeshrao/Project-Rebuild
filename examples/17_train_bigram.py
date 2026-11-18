"""Lesson 17: train a bigram model end-to-end on our tiny character corpus."""

import torch

from rebuild_gpt.bigram import BigramLanguageModel
from rebuild_gpt.data import load_corpus, sample_batch, split_train_val
from rebuild_gpt.training import train_step

torch.manual_seed(3)
tokens, tokenizer = load_corpus("data/tiny_story.txt")
train, val = split_train_val(tokens, val_fraction=0.2)
model = BigramLanguageModel(vocab_size=tokenizer.vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.08)

# Fix one held-out mini-batch so before/after are directly comparable.
val_x, val_y = sample_batch(
    val, batch_size=16, block_size=4, generator=torch.Generator().manual_seed(9)
)

def validation_loss() -> float:
    model.eval()
    with torch.no_grad():
        return model.loss(val_x, val_y).item()

before = validation_loss()
training_rng = torch.Generator().manual_seed(7)

for step in range(120):
    x, y = sample_batch(train, batch_size=16, block_size=8, generator=training_rng)
    training_loss = train_step(model, optimizer, x, y)
    if (step + 1) % 40 == 0:
        print(f"step {step + 1:>3}: training batch loss = {training_loss:.4f}")

after = validation_loss()
print(f"held-out loss: {before:.4f} before -> {after:.4f} after")
print("Small validation sets can fluctuate; improvement is not guaranteed.")

prompt = torch.tensor([[tokenizer.stoi["a"]]], dtype=torch.long)
generated = model.generate(
    prompt, 80, generator=torch.Generator().manual_seed(11)
)
print("generated text:")
print(tokenizer.decode(generated[0].tolist()))
