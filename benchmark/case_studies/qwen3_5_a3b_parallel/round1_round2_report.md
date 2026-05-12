# TP2 / DP2 / DP2+EP Two-Round Report

## Scope

- Model: `Qwen3.5-35B-A3B-GPTQ-Int4`
- Round 1: full grid sweep (`concurrency=[4,8,16,32]`, `input_tokens=[256,2048,4096,8192]`)
- Round 2: anomaly-point retest (`concurrency=16`, `input_tokens=256`, 5 runs per mode)

Source data:
- `results/tables/Qwen3.5-35B-A3B-GPTQ-Int4/compare_tp2_dp2_dp2ep.json`
- `results/tables/Qwen3.5-35B-A3B-GPTQ-Int4/retest_c16_in256/retest_compare_summary.json`

Figures:
- `results/figures/Qwen3.5-35B-A3B-GPTQ-Int4/compare/dp2_vs_tp2_heatmaps.png`
- `results/figures/Qwen3.5-35B-A3B-GPTQ-Int4/compare/dp2ep_vs_tp2_heatmaps.png`
- `results/figures/Qwen3.5-35B-A3B-GPTQ-Int4/compare/dp2ep_vs_dp2_heatmaps.png`
- `results/figures/Qwen3.5-35B-A3B-GPTQ-Int4/compare/retest_c16_in256_round1_vs_round2.png`

## Round 1 Summary (Full Sweep)

### DP2 vs TP2 (16 points)
- TTFT delta mean: `-41.72%` (median `-43.26%`)
- Latency delta mean: `-17.93%` (median `-18.18%`)
- Throughput delta mean: `+43.56%` (median `+41.10%`)

Interpretation:
- DP2 is substantially better than TP2 overall, especially in high-load regions.

### DP2+EP vs TP2 (16 points)
- TTFT delta mean: `-43.37%` (median `-49.12%`)
- Latency delta mean: `-20.31%` (median `-19.43%`)
- Throughput delta mean: `+43.94%` (median `+20.59%`)

Interpretation:
- DP2+EP is also clearly better than TP2 overall.

### DP2+EP vs DP2 (16 points)
- TTFT delta mean: `+12.09%` (median `-2.22%`)
- Latency delta mean: `-3.05%` (median `-3.39%`)
- Throughput delta mean: `+0.92%` (median `+3.56%`)

Interpretation:
- Compared to DP2, DP2+EP is mixed but slightly favorable on median latency/throughput.
- Mean TTFT is skewed by outliers.

## Round 1 Anomaly Point

Point: `concurrency=16`, `input_tokens=256`

From Round 1:
- DP2: `ttft=0.625`, `latency=36.93`, `throughput=58.95`
- DP2+EP: `ttft=2.139`, `latency=37.00`, `throughput=40.65`
- DP2+EP vs DP2 delta:
  - TTFT: `+242.03%`
  - Latency: `+0.19%`
  - Throughput: `-31.05%`

This was flagged as abnormal and triggered Round 2 retest.

## Round 2 Retest (5 runs each)

Retest point: `concurrency=16`, `input_tokens=256`

Median over 5 runs:
- DP2:
  - TTFT: `0.6181`
  - Latency: `34.7533`
  - Throughput: `62.6266`
- DP2+EP:
  - TTFT: `0.5803`
  - Latency: `35.5304`
  - Throughput: `61.1273`

Round 2 median delta (DP2+EP vs DP2):
- TTFT: `-6.11%`
- Latency: `+2.24%`
- Throughput: `-2.39%`

Interpretation:
- The Round 1 extreme anomaly (`TTFT +242%`, throughput `-31%`) was **not reproduced**.
- Round 2 indicates DP2 and DP2+EP are close at this point.
- DP2+EP is slightly better in TTFT but slightly worse in latency/throughput for this specific configuration.

## Final Conclusion

1. At system level, both DP2 and DP2+EP outperform TP2 on this workload set.
2. The previously observed severe regression at `(c=16, tokens=256)` appears to be a one-off/outlier.
3. For this anomaly point, the retest shows no catastrophic DP2+EP degradation; differences are small and mixed.
4. Decision guidance:
   - If you optimize for robust aggregate improvement over TP2, DP2 family remains preferable.
   - Between DP2 and DP2+EP, choose based on broader repeated runs and your primary KPI (TTFT vs throughput).
