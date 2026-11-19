"""Lesson 225: train a tiny GPT and save its model and run evidence."""

from rebuild_gpt.capstone import save_capstone_run, train_capstone
from rebuild_gpt.capstone_spec import CapstoneSpec


def main() -> None:
    spec = CapstoneSpec(steps=3)
    corpus = "data/tiny_story.txt"
    model, tokenizer, metrics = train_capstone(corpus, spec)
    report = save_capstone_run(model, tokenizer, metrics, corpus, "out/capstone", spec)
    print("saved: out/capstone/model.pt")
    print("saved: out/capstone/run.json")
    print("saved: out/capstone/sample.txt")
    print("sample:", repr(report["generated_text"]))


if __name__ == "__main__":
    main()
