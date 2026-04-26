# Profiling Analysis

This folder contains the post-processing pipeline for profiling outputs.

## Scope

- `profiling/torch/*`: parse and summarize PyTorch profiler JSON traces.
- `profiling/nsys/*`: parse and summarize NSight Systems exports (`.sqlite/.csv/.txt`).
- `profiling/run_pipeline.py`: one-command pipeline for both tracks.
- `profiling/classify_traces.py`: trace inventory and classification.
- `profiling/TRACING_WORKFLOW.md`: tracing methods and standard workflow.

## Recommended Workflow

1. Produce traces from your benchmark/profiling run:
- Torch profiler trace JSON in `results/traces/torch/`
- NSYS export (`.sqlite`) in `results/traces/`

2. (Optional but recommended) classify traces:

```bash
python benchmark/analysis/profiling/classify_traces.py
```

2.1 (Optional) reorganize traces to standard layout:

```bash
python benchmark/analysis/profiling/reorganize_traces.py
```

3. Run full post-analysis:

```bash
python benchmark/analysis/profiling/run_pipeline.py \
  --torch-trace-dir results/traces/torch \
  --nsys-trace-dir results/traces/nsys \
  --skip-missing
```

4. Check outputs:
- `results/analysis/profiling/torch/`
- `results/analysis/profiling/nsys/`
- `results/figures/profiling/torch/`
- `results/figures/profiling/nsys/`

## Notes

- `run_pipeline.py` analyzes traces only and does not perform trace collection.
- Trace collection commands and conventions are documented in:
  - `benchmark/analysis/profiling/TRACING_WORKFLOW.md`
