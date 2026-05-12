# Qwen3-8B Dense FX Graphs

This folder is reserved for FX/operator artifacts for the primary Qwen3-8B dense optimization project.

## Planned Contents

- `config.yaml`
Model-specific FX metadata and expected output paths.

- `fx_graphs/`
Layer-level exported graph files for Qwen3-8B dense blocks.

- `artifacts/`
Model graph and module listing artifacts.

- `docs/`
Operator analysis notes for Qwen3-8B dense.

## Next Step

Generate FX graphs after the dual-card baseline is fixed, so operator analysis is tied to the same model and serving target as the main benchmark project.
