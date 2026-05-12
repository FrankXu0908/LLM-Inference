# 1. Model Architecture and Operator Analysis

## Overview

This document records operator-level notes for the Qwen3.5-35B-A3B FX graph artifacts in this folder.

## Qwen3.5-35B-A3B Architecture Notes

### Model Specifications

- **Model**: Qwen3.5-35B-A3B-GPTQ-Int4
- **Layers**: 40 transformer layers
- **Architecture**: hybrid attention / MoE-style block structure
- **Artifact root**: `fx/models/qwen3_5_35b_a3b/`

### Key Components

#### 1. Embedding Layer
- **Purpose**: Convert tokens to vector representations
- **Operation**: Lookup table (vocab_size × hidden_size)
- **Compute**: Negligible compared to transformer layers
- **Memory**: ~600MB for embeddings (FP16)

#### 2. Transformer Layers (×40)

Each layer consists of:

##### Multi-Head Attention
- **Input**: Hidden states + positional embeddings
- **Operations**:
  - QKV projection: Linear(hidden_size → 3×hidden_size)
  - Attention computation: softmax(QK^T/√d) × V
  - Output projection: Linear(hidden_size → hidden_size)
- **Compute Complexity**: O(seq_len² × hidden_size)
- **Memory Access**: High bandwidth for attention matrix

##### Feed-Forward Network (FFN)
- **Operations**:
  - Up projection: Linear(hidden_size → intermediate_size)
  - Activation: SiLU (Swish)
  - Down projection: Linear(intermediate_size → hidden_size)
- **Compute Complexity**: O(seq_len × intermediate_size)
- **Bottleneck**: Large intermediate size (11008) creates memory pressure

##### Layer Normalization
- **Operations**: RMSNorm (simplified LayerNorm)
- **Compute**: Lightweight, O(seq_len × hidden_size)
- **Purpose**: Stabilize training, normalize activations

#### 3. Language Modeling Head
- **Operation**: Linear(hidden_size → vocab_size)
- **Compute**: O(seq_len × vocab_size)
- **Memory**: Large weight matrix (~3GB for FP16)

## Operator-Level Analysis

### Computational Operators

#### 1. Matrix Multiplication (GEMM)
- **Dominant Operation**: ~70-80% of total compute
- **Locations**:
  - Attention QKV projections
  - Attention output projection
  - FFN up/down projections
  - Language modeling head
- **Characteristics**:
  - Memory bandwidth bound
  - Tensor core utilization critical
  - Batch size affects efficiency

#### 2. Element-wise Operations
- **Operations**: SiLU, softmax, addition, multiplication
- **Locations**: FFN activation, attention softmax, residual connections
- **Characteristics**:
  - Compute bound
  - High parallelism
  - Low memory bandwidth requirements

#### 3. Reduction Operations
- **Operations**: sum, max (for softmax)
- **Locations**: Attention normalization, layer norm
- **Characteristics**:
  - Memory latency bound
  - Sequential dependencies
  - Optimization opportunities limited

### Memory Access Patterns

#### KV Cache Management
- **Purpose**: Store key/value vectors for autoregressive generation
- **Size**: seq_len × num_kv_heads × head_dim × 2 (K+V)
- **Access Pattern**: Sequential writes, random reads during decode
- **Optimization**: Paged attention, quantization

#### Weight Matrices
- **Storage**: FP16/BF16 precision
- **Access**: Regular patterns during inference
- **Optimization**: Quantization (GPTQ, AWQ), sparsity

## Performance Characteristics

### Compute Distribution

```
Component          | FLOPs (%) | Memory Access (%) | Notes
-------------------|-----------|-------------------|-------
Attention          | 45-55     | 60-70            | O(n²) scaling
FFN               | 35-45     | 20-30            | Memory bandwidth
Layer Norm        | <1        | <1               | Negligible
Embedding         | <1        | 5-10             | Initial access
LM Head           | 5-10      | 5-10             | Final projection
```

### Bottlenecks by Phase

#### Prefill Phase (Input Processing)
- **Bottleneck**: Attention computation (O(n²))
- **Optimization**: Flash Attention, memory-efficient attention
- **Hardware**: Tensor cores for GEMM, high memory bandwidth

#### Decode Phase (Token Generation)
- **Bottleneck**: Memory latency for KV cache access
- **Optimization**: Paged attention, KV cache quantization
- **Hardware**: High memory bandwidth, fast HBM

## Optimization Opportunities

### Operator Fusion
- **Linear + Activation**: Fuse bias addition and activation
- **Attention Operations**: Fuse QKV projection with attention
- **Layer Norm**: Fuse with adjacent linear operations

### Precision Optimization
- **FP16/BF16**: Standard for inference
- **INT8 Quantization**: 2x memory reduction, potential speedup
- **Dynamic Scaling**: Adaptive precision based on layer importance

### Memory Optimization
- **KV Cache**: Quantization, compression, offloading
- **Activation Checkpointing**: Trade compute for memory
- **Weight Quantization**: Reduce model size

## Profiling Insights

### Key Metrics to Monitor

1. **Operator Timing**: Time spent in each operation type
2. **Memory Bandwidth**: Peak utilization during attention
3. **Tensor Core Utilization**: Efficiency of matrix operations
4. **Kernel Launch Overhead**: Frequency of small operations

### Profiling Commands

```bash
# PyTorch profiler
with torch.profiler.profile(...) as prof:
    model(input_ids)

# NSight Systems
nsys profile --trace=cuda,nvtx python script.py

# Custom profiling
profile_operators(model, input_data)
```

## Architecture Evolution

### Trends in LLM Design

1. **Grouped Query Attention (GQA)**: Reduces KV cache size
2. **Multi-Query Attention (MQA)**: Further KV cache reduction
3. **Sliding Window Attention**: Limited context for efficiency
4. **Mixture of Experts (MoE)**: Conditional computation

### Hardware Considerations

- **GPU Memory**: Limits maximum sequence length
- **Tensor Cores**: Accelerate matrix operations
- **Memory Bandwidth**: Critical for attention computation
- **Interconnect**: Important for multi-GPU setups

## Conclusion

Understanding the operator-level characteristics of Qwen3.5-35B-A3B provides background for the secondary A3B case study. The primary optimization target of the repo is now Qwen3-8B dense; use this document as supporting operator context rather than the main project plan.
