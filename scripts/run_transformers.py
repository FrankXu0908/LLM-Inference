from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

"""
nsys profile \
  --trace=cuda \
  -o trace_transformers \
  python run_transformers.py
  """
config = load_config("configs/model.yaml")
model_name = config["model"]["name"]

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             dtype=config["model"]["dtype"],
                                             device_map="cuda",).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer("我想去马尔代夫旅行，帮我规划一个行程并且避坑", return_tensors="pt").to("cuda")

with torch.no_grad():
    for _ in range(10):
        outputs = model(**inputs)