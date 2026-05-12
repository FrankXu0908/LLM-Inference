#!/usr/bin/env python3
"""
Operator Statistics and Analysis

This script provides comprehensive statistics about operators
in LLM models, including frequency, computational cost,
memory usage, and optimization opportunities.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_mapping_results(mapping_dir: str) -> Dict[str, Any]:
    """Load operator mapping results."""
    mapping_path = Path(mapping_dir)

    with open(mapping_path / "operator_kernel_mapping.json", 'r') as f:
        mapped_operations = json.load(f)

    with open(mapping_path / "kernel_summary.json", 'r') as f:
        kernel_summary = json.load(f)

    return {
        'mapped_operations': mapped_operations,
        'kernel_summary': kernel_summary
    }

def compute_detailed_statistics(mapped_operations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute detailed operator statistics."""
    print("Computing detailed operator statistics...")

    stats = {
        'operation_counts': Counter(),
        'flops_distribution': defaultdict(float),
        'memory_usage': defaultdict(float),
        'kernel_types': Counter(),
        'fusion_potential': defaultdict(int),
        'shape_analysis': defaultdict(list),
        'parameter_counts': defaultdict(int)
    }

    for op in mapped_operations:
        # Basic counts
        op_name = op['module_name']
        stats['operation_counts'][op_name] += 1
        stats['kernel_types'][op['kernel_type']] += 1

        # FLOPs
        flops = op.get('flops', 0)
        stats['flops_distribution'][op_name] += flops

        # Memory usage estimation
        if 'input_shape' in op and op['input_shape']:
            input_size = estimate_tensor_size(op['input_shape'])
            stats['memory_usage'][op_name] += input_size

        if 'output_shape' in op:
            output_size = estimate_tensor_size([op['output_shape']])
            stats['memory_usage'][op_name] += output_size

        # Parameters
        if 'parameters' in op:
            param_count = sum(np.prod(shape) for shape in op['parameters'].values())
            stats['parameter_counts'][op_name] += param_count

        # Fusion potential
        fusion_ops = op.get('fusion_potential', {}).get('fusion_opportunities', [])
        for fusion in fusion_ops:
            stats['fusion_potential'][fusion] += 1

        # Shape analysis
        if 'input_shape' in op:
            stats['shape_analysis'][op_name].extend(op['input_shape'])

    return stats

def estimate_tensor_size(shapes: List) -> float:
    """Estimate tensor size in bytes."""
    total_size = 0
    for shape in shapes:
        if isinstance(shape, list) and len(shape) > 0:
            # Assume float16 (2 bytes per element)
            size = np.prod(shape) * 2
            total_size += size
    return total_size

def create_statistics_visualizations(stats: Dict[str, Any], output_dir: str):
    """Create comprehensive statistics visualizations."""
    print("Creating statistics visualizations...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Operation frequency bar chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Operation counts
    ops = list(stats['operation_counts'].keys())
    counts = list(stats['operation_counts'].values())

    bars1 = ax1.bar(range(len(ops)), counts, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Operation Type')
    ax1.set_ylabel('Count')
    ax1.set_title('Operation Frequency')
    ax1.set_xticks(range(len(ops)))
    ax1.set_xticklabels(ops, rotation=45, ha='right')

    # Add value labels
    for bar, count in zip(bars1, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{count}', ha='center', va='bottom', fontsize=8)

    # 2. FLOPs distribution
    flops_ops = list(stats['flops_distribution'].keys())
    flops_values = list(stats['flops_distribution'].values())

    bars2 = ax2.bar(range(len(flops_ops)), flops_values, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Operation Type')
    ax2.set_ylabel('FLOPs')
    ax2.set_title('FLOPs Distribution')
    ax2.set_xticks(range(len(flops_ops)))
    ax2.set_xticklabels(flops_ops, rotation=45, ha='right')
    ax2.set_yscale('log')

    # 3. Memory usage
    mem_ops = list(stats['memory_usage'].keys())
    mem_values = list(stats['memory_usage'].values())

    bars3 = ax3.bar(range(len(mem_ops)), mem_values, color='lightgreen', edgecolor='black')
    ax3.set_xlabel('Operation Type')
    ax3.set_ylabel('Memory Usage (bytes)')
    ax3.set_title('Memory Usage Distribution')
    ax3.set_xticks(range(len(mem_ops)))
    ax3.set_xticklabels(mem_ops, rotation=45, ha='right')
    ax3.set_yscale('log')

    # 4. Kernel type distribution (pie chart)
    kernel_types = list(stats['kernel_types'].keys())
    kernel_counts = list(stats['kernel_types'].values())

    ax4.pie(kernel_counts, labels=kernel_types, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Kernel Type Distribution')
    ax4.axis('equal')

    plt.tight_layout()
    plt.savefig(output_path / "operator_statistics.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Fusion potential analysis
    if stats['fusion_potential']:
        fig, ax = plt.subplots(figsize=(10, 6))

        fusions = list(stats['fusion_potential'].keys())
        fusion_counts = list(stats['fusion_potential'].values())

        bars = ax.bar(fusions, fusion_counts, color='orange', edgecolor='black')
        ax.set_xlabel('Fusion Type')
        ax.set_ylabel('Count')
        ax.set_title('Fusion Opportunities')
        ax.tick_params(axis='x', rotation=45, ha='right')

        for bar, count in zip(bars, fusion_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{count}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path / "fusion_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Visualizations saved to {output_path}")

def generate_optimization_recommendations(stats: Dict[str, Any]) -> List[str]:
    """Generate optimization recommendations based on statistics."""
    recommendations = []

    # High FLOPs operations
    total_flops = sum(stats['flops_distribution'].values())
    high_flops_ops = [(op, flops) for op, flops in stats['flops_distribution'].items()
                     if flops > total_flops * 0.1]  # Top 10%

    if high_flops_ops:
        recommendations.append("High FLOPs Operations to Optimize:")
        for op, flops in sorted(high_flops_ops, key=lambda x: x[1], reverse=True):
            recommendations.append(f"  - {op}: {flops:,} FLOPs ({flops/total_flops*100:.1f}%)")

    # Fusion opportunities
    if stats['fusion_potential']:
        total_fusions = sum(stats['fusion_potential'].values())
        recommendations.append(f"\nFusion Opportunities: {total_fusions} potential fusions")

        for fusion, count in sorted(stats['fusion_potential'].items(), key=lambda x: x[1], reverse=True):
            recommendations.append(f"  - {fusion}: {count} instances")

    # Memory-intensive operations
    total_memory = sum(stats['memory_usage'].values())
    high_mem_ops = [(op, mem) for op, mem in stats['memory_usage'].items()
                   if mem > total_memory * 0.1]  # Top 10%

    if high_mem_ops:
        recommendations.append("\nMemory-Intensive Operations:")
        for op, mem in sorted(high_mem_ops, key=lambda x: x[1], reverse=True):
            recommendations.append(f"  - {op}: {mem:,} bytes ({mem/total_memory*100:.1f}%)")

    # Kernel optimization suggestions
    kernel_dist = stats['kernel_types']
    if kernel_dist.get('linear', 0) > kernel_dist.get('attention', 0):
        recommendations.append("\nConsider kernel fusion for linear operations")
    if kernel_dist.get('attention', 0) > 0:
        recommendations.append("Evaluate Flash Attention implementation")

    return recommendations

def save_statistics_report(stats: Dict[str, Any], recommendations: List[str], output_dir: str):
    """Save comprehensive statistics report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "operator_statistics_report.txt", "w") as f:
        f.write("Operator Statistics and Analysis Report\n")
        f.write("=" * 50 + "\n\n")

        f.write("Summary Statistics:\n")
        f.write(f"  Total Operations: {sum(stats['operation_counts'].values())}\n")
        f.write(f"  Unique Operation Types: {len(stats['operation_counts'])}\n")
        f.write(f"  Total FLOPs: {sum(stats['flops_distribution'].values()):,}\n")
        f.write(f"  Total Memory Usage: {sum(stats['memory_usage'].values()):,} bytes\n")
        f.write(f"  Kernel Types: {len(stats['kernel_types'])}\n\n")

        f.write("Operation Frequency:\n")
        for op, count in sorted(stats['operation_counts'].items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {op}: {count}\n")

        f.write("\nFLOPs Distribution:\n")
        total_flops = sum(stats['flops_distribution'].values())
        for op, flops in sorted(stats['flops_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = flops / total_flops * 100 if total_flops > 0 else 0
            f.write(f"  {op}: {flops:,} ({percentage:.1f}%)\n")

        f.write("\nOptimization Recommendations:\n")
        for rec in recommendations:
            f.write(f"{rec}\n")

    # Save as CSV for further analysis
    df_data = []
    for op in stats['operation_counts'].keys():
        df_data.append({
            'operation': op,
            'count': stats['operation_counts'][op],
            'flops': stats['flops_distribution'][op],
            'memory': stats['memory_usage'][op],
            'parameters': stats['parameter_counts'][op]
        })

    df = pd.DataFrame(df_data)
    df.to_csv(output_path / "operator_statistics.csv", index=False)

    print(f"Statistics report saved to {output_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Operator Statistics Analysis")
    parser.add_argument(
        "--mapping-dir",
        default="fx/models/qwen3_8b_dense/artifacts/operator_mapping",
        help="Mapping results directory",
    )
    parser.add_argument(
        "--output-dir",
        default="fx/models/qwen3_8b_dense/artifacts/operator_stats",
        help="Output directory",
    )

    args = parser.parse_args()

    # Load mapping results
    mapping_results = load_mapping_results(args.mapping_dir)

    # Compute detailed statistics
    stats = compute_detailed_statistics(mapping_results['mapped_operations'])

    # Generate recommendations
    recommendations = generate_optimization_recommendations(stats)

    # Print summary
    print("\nOperator Statistics Summary:")
    print(f"Total operations: {sum(stats['operation_counts'].values())}")
    print(f"Total FLOPs: {sum(stats['flops_distribution'].values()):,}")
    print(f"Total memory: {sum(stats['memory_usage'].values()):,} bytes")
    print(f"Kernel types: {len(stats['kernel_types'])}")

    # Create visualizations
    create_statistics_visualizations(stats, str(Path(args.output_dir) / "figures"))

    # Save report
    save_statistics_report(stats, recommendations, args.output_dir)

    print("Statistics analysis completed!")

if __name__ == "__main__":
    main()
