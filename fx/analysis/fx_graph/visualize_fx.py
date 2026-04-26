#!/usr/bin/env python3
"""
Visualize FX Graphs and Computation Patterns

This script provides advanced visualization capabilities for FX graphs,
including operator fusion analysis, memory access patterns, and
computation flow diagrams.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import numpy as np

def load_fx_analysis(analysis_dir: str) -> Dict[str, Any]:
    """Load FX graph analysis results."""
    analysis_path = Path(analysis_dir)

    # Load operations
    with open(analysis_path / "operations.json", 'r') as f:
        operations = json.load(f)

    # Load dependency graph
    dep_graph = nx.read_graphml(analysis_path / "dependency_graph.graphml")

    return {
        'operations': operations,
        'dependency_graph': dep_graph
    }

def categorize_operations(operations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize operations by type and functionality."""
    categories = defaultdict(list)

    for op in operations:
        if op['op'] == 'call_function':
            target = op['target']
            if 'matmul' in target or 'linear' in target:
                categories['linear'].append(op)
            elif 'attention' in target or 'attn' in target.lower():
                categories['attention'].append(op)
            elif 'layernorm' in target or 'norm' in target:
                categories['normalization'].append(op)
            elif 'activation' in target or 'relu' in target or 'gelu' in target:
                categories['activation'].append(op)
            elif 'dropout' in target:
                categories['regularization'].append(op)
            elif 'embedding' in target:
                categories['embedding'].append(op)
            else:
                categories['other'].append(op)
        elif op['op'] == 'call_module':
            categories['module'].append(op)
        else:
            categories['other'].append(op)

    return dict(categories)

def create_operator_flow_diagram(operations: List[Dict[str, Any]], output_path: str):
    """Create a flow diagram showing operator execution order."""
    print("Creating operator flow diagram...")

    categories = categorize_operations(operations)

    # Create a simplified flow graph
    flow_graph = nx.DiGraph()

    # Add nodes for each category
    for category, ops in categories.items():
        flow_graph.add_node(category,
                          size=len(ops),
                          operations=ops,
                          color=get_category_color(category))

    # Add edges based on typical transformer flow
    typical_flow = ['embedding', 'attention', 'normalization', 'activation', 'linear', 'other']
    for i in range(len(typical_flow) - 1):
        if typical_flow[i] in flow_graph and typical_flow[i+1] in flow_graph:
            flow_graph.add_edge(typical_flow[i], typical_flow[i+1])

    # Visualize
    plt.figure(figsize=(12, 8))

    pos = nx.spring_layout(flow_graph, k=2, iterations=50)

    node_sizes = [flow_graph.nodes[node]['size'] * 100 + 500 for node in flow_graph.nodes()]
    node_colors = [flow_graph.nodes[node]['color'] for node in flow_graph.nodes()]

    nx.draw(flow_graph, pos, with_labels=True, node_color=node_colors,
            node_size=node_sizes, font_size=12, font_weight='bold',
            arrows=True, arrowsize=20, edge_color='gray', alpha=0.8)

    # Add operation counts as labels
    labels = {node: f"{node}\n({flow_graph.nodes[node]['size']} ops)" for node in flow_graph.nodes()}
    nx.draw_networkx_labels(flow_graph, pos, labels, font_size=10)

    plt.title("Operator Flow Diagram", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Flow diagram saved to {output_path}")

def get_category_color(category: str) -> str:
    """Get color for operation category."""
    color_map = {
        'embedding': 'lightblue',
        'attention': 'lightgreen',
        'normalization': 'lightyellow',
        'activation': 'lightcoral',
        'linear': 'lightpink',
        'regularization': 'lightgray',
        'module': 'orange',
        'other': 'white'
    }
    return color_map.get(category, 'white')

def create_memory_access_pattern(dep_graph: nx.DiGraph, output_path: str):
    """Analyze and visualize memory access patterns."""
    print("Analyzing memory access patterns...")

    # Calculate node depths (layers)
    try:
        depths = nx.shortest_path_length(dep_graph, source=list(dep_graph.nodes())[0])
    except:
        depths = {node: 0 for node in dep_graph.nodes()}

    # Group nodes by depth
    depth_groups = defaultdict(list)
    for node, depth in depths.items():
        depth_groups[depth].append(node)

    # Create visualization
    fig, ax = plt.subplots(figsize=(15, 10))

    max_depth = max(depths.values()) if depths else 0
    y_positions = np.linspace(0.1, 0.9, max_depth + 1)

    for depth, nodes in depth_groups.items():
        x_positions = np.linspace(0.1, 0.9, len(nodes))
        colors = [get_node_color(dep_graph.nodes[node].get('op', 'other')) for node in nodes]

        ax.scatter(x_positions, [y_positions[depth]] * len(nodes),
                  c=colors, s=100, alpha=0.7, edgecolors='black')

        # Add labels for important nodes
        for i, node in enumerate(nodes):
            if i % max(1, len(nodes) // 10) == 0:  # Label every 10th node or so
                ax.annotate(node[:10], (x_positions[i], y_positions[depth]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_xlabel('Node Position')
    ax.set_ylabel('Computation Depth')
    ax.set_title('Memory Access Pattern Analysis', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='call_function'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='call_method'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=10, label='call_module'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Memory access pattern saved to {output_path}")

def get_node_color(op: str) -> str:
    """Get color for node operation type."""
    color_map = {
        'call_function': 'lightblue',
        'call_method': 'lightgreen',
        'call_module': 'lightcoral',
        'other': 'lightgray'
    }
    return color_map.get(op, 'lightgray')

def create_operator_statistics(categories: Dict[str, List[Dict[str, Any]]], output_path: str):
    """Create operator statistics visualization."""
    print("Creating operator statistics...")

    # Count operations per category
    category_counts = {cat: len(ops) for cat, ops in categories.items()}

    # Create bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar chart
    categories_list = list(category_counts.keys())
    counts = list(category_counts.values())

    bars = ax1.bar(categories_list, counts, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Operation Category')
    ax1.set_ylabel('Count')
    ax1.set_title('Operations by Category')
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')

    # Pie chart
    ax2.pie(counts, labels=categories_list, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Operation Distribution')
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Operator statistics saved to {output_path}")

def save_visualization_summary(categories: Dict[str, List[Dict[str, Any]]], output_dir: str):
    """Save summary of visualizations."""
    summary_path = Path(output_dir) / "visualization_summary.txt"

    with open(summary_path, 'w') as f:
        f.write("FX Graph Visualization Summary\n")
        f.write("=" * 40 + "\n\n")

        f.write("Operation Categories:\n")
        for category, ops in categories.items():
            f.write(f"  {category}: {len(ops)} operations\n")

        f.write("\nGenerated Visualizations:\n")
        f.write("  - operator_flow_diagram.png: Operator execution flow\n")
        f.write("  - memory_access_pattern.png: Memory access patterns by depth\n")
        f.write("  - operator_statistics.png: Operation statistics and distribution\n")

    print(f"Summary saved to {summary_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize FX Graphs")
    parser.add_argument("--analysis-dir", default="results/fx_graph", help="FX analysis directory")
    parser.add_argument("--output-dir", default="results/figures", help="Output directory")

    args = parser.parse_args()

    # Load analysis
    analysis = load_fx_analysis(args.analysis_dir)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Categorize operations
    categories = categorize_operations(analysis['operations'])

    # Create visualizations
    create_operator_flow_diagram(analysis['operations'], output_path / "operator_flow_diagram.png")
    create_memory_access_pattern(analysis['dependency_graph'], output_path / "memory_access_pattern.png")
    create_operator_statistics(categories, output_path / "operator_statistics.png")

    # Save summary
    save_visualization_summary(categories, str(output_path))

    print("Visualization completed!")

if __name__ == "__main__":
    main()