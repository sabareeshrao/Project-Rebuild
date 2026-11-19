# Lesson 152 — Turn pipeline timings into testable questions

Stage medians alone are useful but hard to compare across many model
shapes. A small analysis helper can identify the largest stage for
each row of our pipeline CSV experiments.

```bash
python examples/102_pipeline_scaling_sweep.py \
  --batch-sizes 1,2,4 --depths 1,2 --csv out/pipeline.csv
python examples/103_analyze_pipeline.py \
  --input out/pipeline.csv --output out/pipeline_analysis.json
```

The analyzer chooses the highest of sampling, transfer and optimizer
median time, then suggests **what to investigate next**. It does not
automatically optimize anything or promise a speedup.

**Statistical caution:** The median of a sum is not generally equal to
the sum of individual medians. We therefore label each fraction as
a share of the **sum of stage medians**, *not* a fraction of the
end-to-end median. No CUDA transfer/compute overlap can be inferred:
our stage timing deliberately synchronizes between stages.

A CPU transfer-stage measurement is CPU tensor handoff/copy overhead,
not a measured host-to-CUDA transfer.

Run `python -m pytest tests/test_pipeline_analysis.py -q`.

**Experiment:** Increase n_layer and see whether the largest median
stage changes. Try a much larger batch with the same model.

**Checkpoint:** Why is an observed slowest-stage label a starting
point for profiling rather than proof of what optimization will help?
