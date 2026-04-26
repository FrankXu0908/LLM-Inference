import yaml
from typing import Dict, Any
from vllm import LLM, SamplingParams

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

"""
nsys profile \
  --trace=cuda \
  --force-overwrite true \
  -o trace_vllm \
  python scripts/run_vllm.py
  """

if __name__ == "__main__":
    config = load_config("configs/model.yaml")
    model_name = config["model"]["name"]

    model = LLM(model_name,
                dtype=config["model"]["dtype"],
                tensor_parallel_size=config["model"]["tensor_parallel_size"],
                max_model_len=config["model"]["max_model_len"],
                enable_prefix_caching=False,)

    inputs = ["我想去马尔代夫旅行，帮我规划一个行程并且避坑"]


    for _ in range(5):
        outputs = model.generate(inputs, SamplingParams(max_tokens=1024, temperature=0.7, top_p=0.9))