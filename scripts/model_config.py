from pathlib import Path

import yaml


def load_model_name(config_path: str = "configs/model.yaml") -> str:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    model = ((cfg.get("model") or {}).get("name") or "").strip()
    if not model:
        raise ValueError(f"`model.name` is missing in {path}")
    return model
