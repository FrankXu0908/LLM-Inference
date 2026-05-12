import os
import torch
import torch.nn as nn
from torch.export import export

class SelfAttnWrapper(nn.Module):
    def __init__(self, attn_module, rotary_emb):
        super().__init__()
        self.attn = attn_module
        self.rotary_emb = rotary_emb

    def forward(self, hidden_states):
        B, T, D = hidden_states.shape

        position_ids = torch.arange(T, device=hidden_states.device).unsqueeze(0).expand(B, -1)

        # 直接让模型自己的 rotary_emb 产生正确格式的 position_embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        attention_mask = torch.zeros(B, 1, T, T, device=hidden_states.device)

        return self.attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            use_cache=False,
        )

def make_wrapper(module):
    """Wrap module to standard forward signature"""
    class Wrapper(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states):
            return self.module(hidden_states)

    return Wrapper(module).to("meta").eval()


def export_submodule(module, input_shape, filename):
    """Export a submodule to FX graph file"""
    try:
        wrapper = make_wrapper(module)
        dummy_input = torch.randn(*input_shape, device="meta")

        gm = export(wrapper, (dummy_input,), strict=False)

        with open(filename, "w") as f:
            f.write(str(gm.graph))

        print(f"[OK] {filename}")

    except Exception as e:
        print(f"[FAIL] {filename}: {e}")


def export_self_attn(layer, rotary_emb, hidden_size, output_file):
    try:
        wrapper = SelfAttnWrapper(
            layer.self_attn,
            rotary_emb,
        ).to("meta").eval()

        dummy_input = torch.randn(1, 16, hidden_size, device="meta")
        gm = export(wrapper, (dummy_input,), strict=False)

        with open(output_file, "w") as f:
            f.write(str(gm.graph))

        print(f"[OK] {output_file}")

    except Exception as e:
        print(f"[FAIL] {output_file}: {e}")
        
def extract_all_layers(model, output_dir="fx/models/qwen3_5_35b_a3b/fx_graphs"):
    os.makedirs(output_dir, exist_ok=True)

    hidden_size = model.config.hidden_size
    input_shape = (1, 16, hidden_size)

    for i, layer in enumerate(model.model.layers):
        if i == 3 or i == 0:  # 只导出前1和4层
            print(f"\n=== Layer {i} ===")

            layer_dir = os.path.join(output_dir, f"layer_{i}")
            os.makedirs(layer_dir, exist_ok=True)

            # ===== 1. linear_attn or self_attn =====
            if hasattr(layer, "linear_attn"):
                export_submodule(
                    layer.linear_attn,
                    input_shape,
                    os.path.join(layer_dir, "linear_attn.txt"),
                )

            if hasattr(layer, "self_attn"):
                export_self_attn(
                    layer,
                    model.model.rotary_emb,
                    hidden_size,
                    os.path.join(layer_dir, "self_attn.txt"),
                )

            # ===== 2. MLP (MoE) =====
            if hasattr(layer, "mlp"):
                export_submodule(
                    layer.mlp,
                    input_shape,
                    os.path.join(layer_dir, "mlp.txt"),
                )

            # ===== 3. Norms =====
            if hasattr(layer, "input_layernorm"):
                export_submodule(
                    layer.input_layernorm,
                    input_shape,
                    os.path.join(layer_dir, "input_layernorm.txt"),
                )

            if hasattr(layer, "post_attention_layernorm"):
                export_submodule(
                    layer.post_attention_layernorm,
                    input_shape,
                    os.path.join(layer_dir, "post_attention_layernorm.txt"),
                )


def main():
    model_name = "/home/xuliren/repo/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="meta",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()

    extract_all_layers(model)


if __name__ == "__main__":
    main()
