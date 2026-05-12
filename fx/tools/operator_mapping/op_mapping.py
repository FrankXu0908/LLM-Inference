#!/usr/bin/env python3
"""
Operator to Kernel Mapping Analysis

This script analyzes the mapping between PyTorch ATen operators
and CUDA kernels, providing insights into kernel fusion opportunities
and optimization potential.
"""

import torch
import torch.fx as fx
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
from pathlib import Path
from typing import Dict, Any, List, Set
from collections import defaultdict, Counter
import re
import json

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_aten_operations(model, input_ids: torch.Tensor) -> List[Dict[str, Any]]:
    """Extract ATen operations from model execution."""
    print("Extracting ATen operations...")

    operations = []

    def trace_hook(module, input, output):
        """Hook to capture operation details."""
        if hasattr(module, '_parameters') and len(module._parameters) > 0:
            # This is a layer with parameters
            op_info = {
                'module_name': module.__class__.__name__,
                'operation': 'forward',
                'input_shape': [list(inp.shape) if hasattr(inp, 'shape') else str(type(inp)) for inp in input],
                'output_shape': list(output.shape) if hasattr(output, 'shape') else str(type(output)),
                'parameters': {name: list(param.shape) for name, param in module.named_parameters()},
                'flops': estimate_flops(module, input, output)
            }
            operations.append(op_info)

    # Register hooks
    hooks = []
    for module in model.modules():
        if len(list(module.parameters())) > 0:  # Only modules with parameters
            hook = module.register_forward_hook(trace_hook)
            hooks.append(hook)

    # Run forward pass
    with torch.no_grad():
        model(input_ids)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return operations

def estimate_flops(module, input, output) -> int:
    """Estimate FLOPs for a module."""
    flops = 0

    if hasattr(module, 'weight') and module.weight is not None:
        # Linear layer: input_size * output_size * batch_size
        if len(input) > 0 and hasattr(input[0], 'shape'):
            batch_size = input[0].shape[0] if len(input[0].shape) > 0 else 1
            in_features = input[0].shape[-1] if len(input[0].shape) > 1 else module.weight.shape[1]
            out_features = module.weight.shape[0]
            flops = batch_size * in_features * out_features * 2  # multiply-add

    elif 'LayerNorm' in module.__class__.__name__:
        # LayerNorm: 2 * num_elements
        if hasattr(output, 'shape'):
            flops = 2 * output.numel()

    elif 'Attention' in module.__class__.__name__ or 'attn' in module.__class__.__name__.lower():
        # Attention: complex, approximate as QK^T + softmax + V
        if len(input) > 0 and hasattr(input[0], 'shape') and len(input[0].shape) >= 3:
            batch_size, seq_len, hidden_size = input[0].shape[:3]
            # Approximate attention FLOPs
            flops = batch_size * seq_len * seq_len * hidden_size * 2

    return flops

def map_to_cuda_kernels(operations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Map operations to CUDA kernels."""
    print("Mapping operations to CUDA kernels...")

    kernel_mapping = {
        'linear': ['cublasSgemm', 'cublasHgemm', 'cutlass_gemm'],
        'layernorm': ['layernorm_kernel', 'fast_layernorm'],
        'attention': ['flash_attention', 'memory_efficient_attention', 'cutlass_attention'],
        'activation': ['gelu_kernel', 'relu_kernel', 'fast_gelu'],
        'embedding': ['embedding_lookup', 'fused_embedding'],
        'softmax': ['fast_softmax', 'flash_softmax'],
        'dropout': ['dropout_kernel', 'fast_dropout']
    }

    mapped_operations = []

    for op in operations:
        module_name = op['module_name'].lower()

        # Determine kernel type
        kernel_type = 'unknown'
        potential_kernels = []

        if 'linear' in module_name:
            kernel_type = 'linear'
            potential_kernels = kernel_mapping['linear']
        elif 'layernorm' in module_name:
            kernel_type = 'layernorm'
            potential_kernels = kernel_mapping['layernorm']
        elif 'attention' in module_name or 'attn' in module_name:
            kernel_type = 'attention'
            potential_kernels = kernel_mapping['attention']
        elif any(act in module_name for act in ['gelu', 'relu', 'sigmoid']):
            kernel_type = 'activation'
            potential_kernels = kernel_mapping['activation']
        elif 'embedding' in module_name:
            kernel_type = 'embedding'
            potential_kernels = kernel_mapping['embedding']
        elif 'dropout' in module_name:
            kernel_type = 'dropout'
            potential_kernels = kernel_mapping['dropout']

        mapped_op = op.copy()
        mapped_op.update({
            'kernel_type': kernel_type,
            'potential_kernels': potential_kernels,
            'fusion_potential': analyze_fusion_potential(op, operations)
        })

        mapped_operations.append(mapped_op)

    return {
        'mapped_operations': mapped_operations,
        'kernel_summary': summarize_kernel_usage(mapped_operations)
    }

def analyze_fusion_potential(op: Dict[str, Any], all_ops: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze fusion potential for an operation."""
    fusion_opportunities = []

    # Common fusion patterns
    if op['kernel_type'] == 'linear':
        # Linear + Activation fusion
        for other_op in all_ops:
            if other_op['kernel_type'] == 'activation' and other_op != op:
                fusion_opportunities.append('linear_activation_fusion')

        # Linear + LayerNorm fusion
        for other_op in all_ops:
            if other_op['kernel_type'] == 'layernorm' and other_op != op:
                fusion_opportunities.append('linear_layernorm_fusion')

    elif op['kernel_type'] == 'attention':
        # Attention + Linear fusion
        for other_op in all_ops:
            if other_op['kernel_type'] == 'linear' and other_op != op:
                fusion_opportunities.append('attention_linear_fusion')

    return {
        'fusion_opportunities': list(set(fusion_opportunities)),
        'fusion_score': len(set(fusion_opportunities))  # Simple score
    }

def summarize_kernel_usage(mapped_operations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize kernel usage statistics."""
    kernel_counts = Counter(op['kernel_type'] for op in mapped_operations)
    total_flops = sum(op.get('flops', 0) for op in mapped_operations)

    # Calculate kernel efficiency
    kernel_efficiency = {}
    for kernel_type, count in kernel_counts.items():
        type_ops = [op for op in mapped_operations if op['kernel_type'] == kernel_type]
        type_flops = sum(op.get('flops', 0) for op in type_ops)
        kernel_efficiency[kernel_type] = {
            'count': count,
            'flops': type_flops,
            'flops_percentage': type_flops / total_flops * 100 if total_flops > 0 else 0
        }

    return {
        'kernel_counts': dict(kernel_counts),
        'total_flops': total_flops,
        'kernel_efficiency': kernel_efficiency,
        'fusion_opportunities': sum(len(op['fusion_potential']['fusion_opportunities']) for op in mapped_operations)
    }

def save_mapping_results(mapping_results: Dict[str, Any], output_dir: str):
    """Save mapping analysis results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed mapping
    with open(output_path / "operator_kernel_mapping.json", "w") as f:
        json.dump(mapping_results['mapped_operations'], f, indent=2)

    # Save kernel summary
    with open(output_path / "kernel_summary.json", "w") as f:
        json.dump(mapping_results['kernel_summary'], f, indent=2)

    # Save human-readable summary
    with open(output_path / "mapping_summary.txt", "w") as f:
        summary = mapping_results['kernel_summary']
        f.write("Operator to Kernel Mapping Summary\n")
        f.write("=" * 40 + "\n\n")

        f.write("Kernel Usage:\n")
        for kernel, count in summary['kernel_counts'].items():
            f.write(f"  {kernel}: {count} operations\n")

        f.write(f"\nTotal FLOPs: {summary['total_flops']:,}\n")
        f.write(f"Fusion Opportunities: {summary['fusion_opportunities']}\n\n")

        f.write("Kernel Efficiency:\n")
        for kernel, stats in summary['kernel_efficiency'].items():
            f.write(f"  {kernel}:\n")
            f.write(f"    Count: {stats['count']}\n")
            f.write(f"    FLOPs: {stats['flops']:,}\n")
            f.write(f"    Percentage: {stats['flops_percentage']:.1f}%\n")

    print(f"Mapping results saved to {output_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Operator to Kernel Mapping Analysis")
    parser.add_argument("--config", default="fx/models/qwen3_8b_dense/config.yaml", help="Model configuration")
    parser.add_argument("--input-text", default="Hello, how are you?", help="Input text for analysis")
    parser.add_argument("--output-dir", default="fx/models/qwen3_8b_dense/artifacts/operator_mapping", help="Output directory")

    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config['model']['name']
    dtype = config['model']['dtype']

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name, dtype)

    # Prepare input
    inputs = tokenizer(args.input_text, return_tensors="pt")
    input_ids = inputs['input_ids'].to(model.device)

    # Extract operations
    operations = extract_aten_operations(model, input_ids)

    # Map to kernels
    mapping_results = map_to_cuda_kernels(operations)

    # Print summary
    summary = mapping_results['kernel_summary']
    print("\nOperator to Kernel Mapping Summary:")
    print(f"Total operations: {len(mapping_results['mapped_operations'])}")
    print(f"Total FLOPs: {summary['total_flops']:,}")
    print(f"Fusion opportunities: {summary['fusion_opportunities']}")
    print("\nKernel distribution:")
    for kernel, count in summary['kernel_counts'].items():
        print(f"  {kernel}: {count}")

    # Save results
    save_mapping_results(mapping_results, args.output_dir)

def load_model_and_tokenizer(model_name: str, dtype: str = "float16"):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype_map.get(dtype, torch.float16),
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    return model, tokenizer

if __name__ == "__main__":
    main()
