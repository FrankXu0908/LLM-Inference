# FX Track

This folder contains model-level FX graph and operator analysis.

## Structure

- `models/qwen3_8b_dense/`
FX workspace for the primary Qwen3-8B dense optimization project.

- `models/qwen3_5_35b_a3b/`
FX artifacts from the secondary Qwen3.5-A3B analysis.

- `tools/`
Generic FX extraction, visualization, and operator mapping scripts.

## Rule

Model-specific outputs should live under `fx/models/<model>/`.

Generic code should live under `fx/tools/`.

## Current Status

- `qwen3_5_35b_a3b`: existing exported graphs and artifacts are archived here.
- `qwen3_8b_dense`: planned FX workspace for the main project.
