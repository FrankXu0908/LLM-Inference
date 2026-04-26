#!/usr/bin/env python3
"""
Extract FX Graphs from LLM Models

This script extracts and analyzes PyTorch FX computation graphs
from transformer models for operator-level analysis.
"""

import torch
import torch.nn as nn
import torch.fx as fx
from torch.export import export
from transformers import AutoConfig, AutoModelForCausalLM
import yaml
from typing import Dict, Any
import networkx as nx
# import matplotlib.pyplot as plt

def load_config(config_path: str) -> Dict[str, Any]:
    """Load model configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_and_tokenizer(model_name: str, dtype: str = "float16"):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    return model

def load_model_no_weights(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="meta",          # ⭐ 核心
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    model.eval()
    return model

def extract_fx_graph(model):
    """Extract FX graph using symbolic tracing."""
    # layers_file_name = "model_layers.txt"
    # with open(layers_file_name, "w") as f:
    #     for name, module in model.named_modules():
    #         f.write(f"{name}: {module}\n")
    block = model.model.layers[0].linear_attn
    class BlockWrapper(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer
            
        def forward(self, hidden_states):
            return self.layer(hidden_states,)
    wrapper = BlockWrapper(block).to("meta").eval()
    
    
    print("Extracting FX graph with symbolic tracing...")
    gm = export(wrapper, (torch.randn(1, 16, model.config.hidden_size, device="meta"),),strict=False)
    
    # # Trace the model
    # gm = fx.symbolic_trace(wrapper)
    # print(gm.graph)
    
    # for node in gm.graph.nodes:
    #     print(node.op, node.target)
    return gm

def analyze_fx_graph(gm: fx.GraphModule) -> Dict[str, Any]:
    """Analyze the FX graph structure."""
    print("Analyzing FX graph...")

    graph = gm.graph

    # Count nodes by type
    node_types = {}
    operations = []

    for node in graph.nodes:
        node_type = type(node).__name__
        node_types[node_type] = node_types.get(node_type, 0) + 1

        if hasattr(node, 'op'):
            operations.append({
                'name': node.name,
                'op': node.op,
                'target': str(node.target) if node.target else None,
                'args': [str(arg) for arg in node.args],
                'kwargs': {k: str(v) for k, v in node.kwargs.items()}
            })

    # Build dependency graph
    dep_graph = nx.DiGraph()

    for node in graph.nodes:
        dep_graph.add_node(node.name, op=node.op, target=str(node.target) if node.target else None)

        for arg in node.args:
            if hasattr(arg, 'name'):
                dep_graph.add_edge(arg.name, node.name)

    analysis = {
        'node_types': node_types,
        'total_nodes': len(graph.nodes),
        'total_operations': len(operations),
        'operations': operations,
        'dependency_graph': dep_graph
    }

    return analysis

# def visualize_fx_graph(analysis: Dict[str, Any], output_path: str):
#     """Visualize the FX graph."""
#     print("Visualizing FX graph...")

#     dep_graph = analysis['dependency_graph']

#     plt.figure(figsize=(20, 16))

#     # Use spring layout for better visualization
#     pos = nx.spring_layout(dep_graph, k=1, iterations=50)

#     # Color nodes by operation type
#     node_colors = []
#     for node in dep_graph.nodes():
#         op = dep_graph.nodes[node]['op']
#         if op == 'call_function':
#             node_colors.append('lightblue')
#         elif op == 'call_method':
#             node_colors.append('lightgreen')
#         elif op == 'call_module':
#             node_colors.append('lightcoral')
#         else:
#             node_colors.append('lightgray')

#     # Draw the graph
#     nx.draw(dep_graph, pos, with_labels=True, node_color=node_colors,
#             node_size=300, font_size=8, font_weight='bold',
#             arrows=True, arrowsize=10, edge_color='gray', alpha=0.7)

#     # Add legend
#     legend_elements = [
#         plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='call_function'),
#         plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='call_method'),
#         plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=10, label='call_module'),
#     ]
#     plt.legend(handles=legend_elements, loc='upper right')

#     plt.title("FX Computation Graph", fontsize=16, fontweight='bold')
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()

#     print(f"Visualization saved to {output_path}")

# def save_fx_analysis(analysis: Dict[str, Any], output_dir: str):
#     """Save FX graph analysis results."""
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)

#     # Save node types
#     with open(output_path / "node_types.txt", "w") as f:
#         f.write("FX Graph Node Types:\n")
#         f.write("-" * 30 + "\n")
#         for node_type, count in analysis['node_types'].items():
#             f.write(f"{node_type}: {count}\n")

#     # Save operations
#     import json
#     with open(output_path / "operations.json", "w") as f:
#         json.dump(analysis['operations'], f, indent=2)

#     # Save dependency graph as GraphML
#     nx.write_graphml(analysis['dependency_graph'], output_path / "dependency_graph.graphml")

#     print(f"Analysis saved to {output_path}")
def layers():
    import argparse

    parser = argparse.ArgumentParser(description="Extract and Analyze FX Graphs")
    parser.add_argument("--config", default="configs/model.yaml", help="Model configuration")
    parser.add_argument("--input-text", default="Hello, how are you?", help="Input text for tracing")
    parser.add_argument("--output-dir", default="results/fx_graph", help="Output directory")

    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config['model']['name']
    dtype = config['model']['dtype']

    # Load model
    model = load_model_and_tokenizer(model_name, dtype)
    # i=0
    # for name, module in model.named_modules():
    #     if "model.layers" in name and i < 4:
    #         print(name, module)
    #         i += 1
    print(len(model.model.layers))
    import inspect
    for i in range(4):
        if i == 0 or i == 3:
            print(model.model.layers[i])
            print(inspect.getsource(model.model.layers[i].forward))
    
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract and Analyze FX Graphs")
    parser.add_argument("--config", default="configs/model.yaml", help="Model configuration")
    parser.add_argument("--input-text", default="Hello, how are you?", help="Input text for tracing")
    parser.add_argument("--output-dir", default="results/fx_graph", help="Output directory")

    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config['model']['name']
    dtype = config['model']['dtype']
    print("dtype:", dtype)
    # Load model
    model = load_model_no_weights(model_name)
    # Extract FX graph
    gm = extract_fx_graph(model)
    output_filename = "model_graph.txt"

    with open(output_filename, "w") as f:
        # gm.graph 通常实现了 __str__ 方法，可以直接写入
        f.write(str(gm.graph))

    
    # print(type(gm))
    # for node in gm.graph.nodes:
    #     print(node.op, node.target)

    # # Analyze graph
    # analysis = analyze_fx_graph(gm)

    # # Print summary
    # print("\nFX Graph Summary:")
    # print(f"Total nodes: {analysis['total_nodes']}")
    # print(f"Total operations: {analysis['total_operations']}")
    # print("Node types:")
    # for node_type, count in analysis['node_types'].items():
    #     print(f"  {node_type}: {count}")

    # # Save results
    # save_fx_analysis(analysis, args.output_dir)

    # # Visualize
    # visualize_fx_graph(analysis, f"{args.output_dir}/fx_graph.png")

if __name__ == "__main__":
    main()