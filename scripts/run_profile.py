#!/usr/bin/env python3
"""
Automated Profiling Pipeline for LLM Inference

This script runs comprehensive profiling of vLLM inference including:
- Torch profiler for operator-level analysis
- NSight Systems for GPU kernel analysis
- Memory profiling
- Custom metrics collection
"""

import os
import sys
import yaml
import time
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_profiling_environment():
    """Setup environment for profiling."""
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def run_torch_profiler(config: Dict[str, Any], model_config: Dict[str, Any]):
    """Run Torch profiler."""
    print("Running Torch Profiler...")

    cmd = [
        sys.executable, "scripts/run_benchmark.py",
        "--config", "configs/profiler.yaml",
        "--server-url", "http://localhost:8000/v1",
        "--output", "/dev/null"  # Discard benchmark output
    ]

    env = os.environ.copy()
    env.update({
        "TORCH_PROFILER_ENABLED": "1",
        "TORCH_PROFILER_OUTPUT_DIR": config["torch_profiler"]["output_dir"],
        "TORCH_PROFILER_RECORD_SHAPES": str(config["torch_profiler"]["record_shapes"]).lower(),
        "TORCH_PROFILER_PROFILE_MEMORY": str(config["torch_profiler"]["profile_memory"]).lower(),
        "TORCH_PROFILER_WITH_STACK": str(config["torch_profiler"]["with_stack"]).lower(),
        "TORCH_PROFILER_WITH_FLOPS": str(config["torch_profiler"]["with_flops"]).lower(),
        "TORCH_PROFILER_WITH_MODULES": str(config["torch_profiler"]["with_modules"]).lower(),
    })

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        print("Torch profiling completed")
        return True
    except subprocess.TimeoutExpired:
        print("Torch profiling timed out")
        return False
    except Exception as e:
        print(f"Torch profiling failed: {e}")
        return False

def run_nsys_profiler(config: Dict[str, Any], model_config: Dict[str, Any]):
    """Run NSight Systems profiler."""
    print("Running NSight Systems Profiler...")

    output_dir = Path(config["nsys_profiler"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    nsys_cmd = [
        "nsys", "profile",
        "--trace", config["nsys_profiler"]["trace"],
        "--sample", config["nsys_profiler"]["sample"],
        "--capture-range", config["nsys_profiler"]["capture_range"],
        "--stats", str(config["nsys_profiler"]["stats"]).lower(),
        "--output", str(output_dir / "profile"),
        sys.executable, "scripts/run_benchmark.py",
        "--config", "configs/profiler.yaml",
        "--server-url", "http://localhost:8000/v1",
        "--output", "/dev/null"
    ]

    try:
        result = subprocess.run(nsys_cmd, capture_output=True, text=True, timeout=600)
        print("NSight profiling completed")
        return True
    except subprocess.TimeoutExpired:
        print("NSight profiling timed out")
        return False
    except FileNotFoundError:
        print("NSight Systems (nsys) not found. Please install CUDA toolkit.")
        return False
    except Exception as e:
        print(f"NSight profiling failed: {e}")
        return False

def run_memory_profiler():
    """Run memory profiling."""
    print("Running Memory Profiler...")

    # Use torch.cuda.memory_summary() and custom hooks
    cmd = [
        sys.executable, "-c",
        """
import torch
import gc
from analysis.profiling.torch.parse_trace import analyze_memory_usage

# Force garbage collection
gc.collect()
torch.cuda.empty_cache()

print('GPU Memory Summary:')
print(torch.cuda.memory_summary())

# Run a quick inference to capture memory usage
# This would be integrated with the actual inference
"""
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("Memory profiling completed")
        with open("results/tables/memory_profile.txt", "w") as f:
            f.write(result.stdout)
        return True
    except Exception as e:
        print(f"Memory profiling failed: {e}")
        return False

def analyze_profiling_results(config: Dict[str, Any]):
    """Analyze profiling results."""
    print("Analyzing profiling results...")

    # Import analysis modules
    try:
        from analysis.profiling.torch.parse_trace import parse_torch_trace
        from analysis.profiling.nsys.parse_nsys import parse_nsys_trace
        from analysis.profiling.torch.summarize import summarize_profiling

        # Parse traces
        if config["torch_profiler"]["enabled"]:
            torch_data = parse_torch_trace(config["torch_profiler"]["output_dir"])
            torch_summary = summarize_profiling(torch_data)
            torch_summary.to_csv("results/tables/torch_profiling_summary.csv")

        if config["nsys_profiler"]["enabled"]:
            nsys_data = parse_nsys_trace(config["nsys_profiler"]["output_dir"])
            nsys_data.to_csv("results/tables/nsys_profiling_summary.csv")

        print("Analysis completed")
        return True

    except ImportError as e:
        print(f"Analysis modules not available: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Automated Profiling Pipeline")
    parser.add_argument("--config", default="configs/profiler.yaml", help="Profiler configuration")
    parser.add_argument("--model-config", default="configs/model.yaml", help="Model configuration")
    parser.add_argument("--torch-only", action="store_true", help="Run only Torch profiler")
    parser.add_argument("--nsys-only", action="store_true", help="Run only NSight profiler")
    parser.add_argument("--memory-only", action="store_true", help="Run only memory profiler")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing traces")

    args = parser.parse_args()

    config = load_config(args.config)
    model_config = load_config(args.model_config)

    setup_profiling_environment()

    if args.analyze_only:
        analyze_profiling_results(config)
        return

    success_count = 0

    if not args.nsys_only and not args.memory_only:
        if run_torch_profiler(config, model_config):
            success_count += 1

    if not args.torch_only and not args.memory_only:
        if run_nsys_profiler(config, model_config):
            success_count += 1

    if not args.torch_only and not args.nsys_only:
        if run_memory_profiler():
            success_count += 1

    if success_count > 0:
        analyze_profiling_results(config)

    print(f"Profiling pipeline completed. {success_count} profilers succeeded.")

if __name__ == "__main__":
    main()