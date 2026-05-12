# Quality Baseline A: DP=1 BF16

This is the quality guardrail for the `Qwen3-8B` dense `DP=1` baseline before applying weight quantization, KV cache FP8, or other inference optimizations.

## Setup

| Item | Value |
|---|---|
| Model | `Qwen3-8B` |
| Model path | `/home/xuliren/repo/models/Qwen/Qwen3-8B` |
| Evaluation stack | `lm_eval` + `vLLM` |
| Parallelism | `TP=1`, `DP=1` |
| dtype | `bfloat16` |
| `max_model_len` | `10000` |
| `gpu_memory_utilization` | `0.85` |
| Prefix caching | enabled |
| Chunked prefill | enabled |
| Batch size | `auto` |
| Few-shot | `0-shot` |

Command:

```bash
lm_eval --model vllm \
  --model_args pretrained=/home/xuliren/repo/models/Qwen/Qwen3-8B,tensor_parallel_size=1,data_parallel_size=1,dtype=bfloat16,max_model_len=10000,gpu_memory_utilization=0.85,enable_prefix_caching=True,enable_chunked_prefill=True \
  --tasks lambada_openai,hellaswag,arc_challenge \
  --batch_size auto \
  --output_path results/eval/qwen3_8b/baseline_a_dp1_bf16/results.json \
  --log_samples \
  2>&1 | tee results/eval/qwen3_8b/baseline_a_dp1_bf16/run.log
```

## Results

| Task | Metric | Value | Stderr | Direction |
|---|---:|---:|---:|---|
| `arc_challenge` | `acc` | `0.5555` | `0.0145` | higher is better |
| `arc_challenge` | `acc_norm` | `0.5640` | `0.0145` | higher is better |
| `hellaswag` | `acc` | `0.5716` | `0.0049` | higher is better |
| `hellaswag` | `acc_norm` | `0.7496` | `0.0043` | higher is better |
| `lambada_openai` | `acc` | `0.6497` | `0.0066` | higher is better |
| `lambada_openai` | `perplexity` | `4.5944` | `0.1387` | lower is better |

Samples:

| Task | Samples |
|---|---:|
| `arc_challenge` | `1172` |
| `hellaswag` | `10042` |
| `lambada_openai` | `5153` |

## Artifacts

- Aggregated result JSON: `results/eval/qwen3_8b/baseline_a_dp1_bf16/results_2026-05-11T23-20-17.119348.json`
- Run log: `results/eval/qwen3_8b/baseline_a_dp1_bf16/run.log`
- Per-sample logs: `results/eval/qwen3_8b/baseline_a_dp1_bf16/samples_*.jsonl`

## Interpretation

This run establishes the quality baseline for the dense BF16 `DP=1` track. Later optimizations should compare against these numbers with the same task set and model arguments.

For quantization experiments, the expected question is not whether serving throughput improves in isolation. The quality gate is whether accuracy/perplexity stays within an acceptable delta while throughput, TTFT, TPOT, or memory behavior improves.
