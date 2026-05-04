from __future__ import annotations

from model.mlp import ModelConfig


def build_mlx_model(input_dim: int, n_classes: int, config: ModelConfig):
    import mlx.nn as mlx_nn

    class MLXResidualBlock(mlx_nn.Module):
        def __init__(self, dim: int, dropout: float):
            super().__init__()
            self.norm = mlx_nn.LayerNorm(dim)
            self.fc1 = mlx_nn.Linear(dim, dim)
            self.fc2 = mlx_nn.Linear(dim, dim)
            self.dropout = mlx_nn.Dropout(p=dropout)

        def __call__(self, x):
            h = self.norm(x)
            h = self.fc1(h)
            h = mlx_nn.gelu(h)
            h = self.dropout(h)
            h = self.fc2(h)
            return x + h

    class MLXResidualMLP(mlx_nn.Module):
        def __init__(self, input_dim: int, n_classes: int, config: ModelConfig):
            super().__init__()
            self.input_proj = mlx_nn.Linear(input_dim, config.hidden_dim)
            self.blocks = [MLXResidualBlock(config.hidden_dim, config.dropout) for _ in range(config.num_blocks)]
            self.head = mlx_nn.Linear(config.hidden_dim, n_classes)

        def __call__(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            return self.head(x)

    return MLXResidualMLP(input_dim, n_classes, config)


def copy_pytorch_weights_to_mlx(mlx_model, state_dict: dict, config: ModelConfig):
    import mlx.core as mx

    def t(key: str):
        return mx.array(state_dict[key].numpy())

    mlx_model.input_proj.weight = t("input_proj.weight")
    mlx_model.input_proj.bias = t("input_proj.bias")
    for i in range(config.num_blocks):
        mlx_model.blocks[i].norm.weight = t(f"blocks.{i}.block.0.weight")
        mlx_model.blocks[i].norm.bias = t(f"blocks.{i}.block.0.bias")
        mlx_model.blocks[i].fc1.weight = t(f"blocks.{i}.block.1.weight")
        mlx_model.blocks[i].fc1.bias = t(f"blocks.{i}.block.1.bias")
        mlx_model.blocks[i].fc2.weight = t(f"blocks.{i}.block.4.weight")
        mlx_model.blocks[i].fc2.bias = t(f"blocks.{i}.block.4.bias")
    mlx_model.head.weight = t("head.weight")
    mlx_model.head.bias = t("head.bias")


def copy_mlx_weights_to_pytorch_sd(state_dict: dict, best_state: dict, config: ModelConfig):
    import numpy as np
    import torch

    def pt(mx_arr):
        return torch.from_numpy(np.array(mx_arr))

    state_dict["input_proj.weight"] = pt(best_state["input_proj"]["weight"])
    state_dict["input_proj.bias"] = pt(best_state["input_proj"]["bias"])
    for i in range(config.num_blocks):
        state_dict[f"blocks.{i}.block.0.weight"] = pt(best_state["blocks"][i]["norm"]["weight"])
        state_dict[f"blocks.{i}.block.0.bias"] = pt(best_state["blocks"][i]["norm"]["bias"])
        state_dict[f"blocks.{i}.block.1.weight"] = pt(best_state["blocks"][i]["fc1"]["weight"])
        state_dict[f"blocks.{i}.block.1.bias"] = pt(best_state["blocks"][i]["fc1"]["bias"])
        state_dict[f"blocks.{i}.block.4.weight"] = pt(best_state["blocks"][i]["fc2"]["weight"])
        state_dict[f"blocks.{i}.block.4.bias"] = pt(best_state["blocks"][i]["fc2"]["bias"])
    state_dict["head.weight"] = pt(best_state["head"]["weight"])
    state_dict["head.bias"] = pt(best_state["head"]["bias"])
