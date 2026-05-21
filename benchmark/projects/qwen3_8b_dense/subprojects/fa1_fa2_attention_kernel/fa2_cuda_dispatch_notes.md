# FA2 CUDA Dispatch Notes

This note records the expected FlashAttention-2 CUDA dispatch for the Qwen3-8B
standalone attention study.

## Target Shape

| Field | Value |
|---|---:|
| GPU | RTX 4090 / Ada / SM89 |
| dtype | BF16 |
| causal | true |
| dropout | false |
| query heads | `32` |
| KV heads | `8` |
| head dim | `128` |

## Source Dispatch

Source root:

```text
/home/xuliren/repo/flash-attention
```

The BF16 causal head-dim 128 entry point is:

```text
csrc/flash_attn/src/flash_fwd_hdim128_bf16_causal_sm80.cu
```

It calls:

```cpp
run_mha_fwd_hdim128<cutlass::bfloat16_t, true>(params, stream);
```

In:

```text
csrc/flash_attn/src/flash_fwd_launch_template.h
```

the SM8x + causal + no-dropout branch dispatches:

```cpp
Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>
```

So for this exact experiment, the expected FA2 CUDA traits are:

| Trait | Value |
|---|---:|
| head dim | `128` |
| block M | `64` |
| block N | `64` |
| warps | `4` |
| Q in registers | `false` |
| shared Q/K smem | `false` |

This corrects the earlier rough assumption of `BM128 BN64 stages4`. For RTX
4090 / SM89 causal head-dim 128, the source chooses `BM64 BN64 warps4`.

## What Nsight Can Verify

Nsight Compute can verify the runtime footprint:

- kernel symbol/name
- grid size
- block size
- registers per thread
- static/dynamic shared memory
- theoretical and achieved occupancy
- waves per SM
- SM/Tensor/DRAM/L2 utilization
- scheduler and stall reasons

Nsight Compute does not reliably expose source template names like `kBlockM`,
`kBlockN`, or `kNWarps` as named fields. Use the source dispatch table as the
ground truth and Nsight as the hardware verification layer.

## First Experiment

Run standalone latency sweep:

```bash
PYTHON_BIN=/home/xuliren/anaconda3/envs/vllm-dev/bin/python \
bash scripts/run_flash_attn_fa2_latency_sweep.sh
```

Run one NCU profile point:

```bash
PYTHON_BIN=/home/xuliren/anaconda3/envs/vllm-dev/bin/python \
BATCH=1 SEQ_LEN=8192 RUN_ID=b1_s8192_run1 \
bash scripts/run_flash_attn_fa2_ncu_profile.sh
```

If `flash_attn` cannot be imported, build the CUDA extension first:

```bash
cd /home/xuliren/repo/flash-attention
pip install -e . --no-build-isolation
```
