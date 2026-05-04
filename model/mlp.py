from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    num_blocks: int = 4
    dropout: float = 0.1
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 96
    max_epochs: int = 600
    early_stopping_patience: int = 40


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, config: ModelConfig):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(config.hidden_dim, config.dropout) for _ in range(config.num_blocks)]
        )
        self.head = nn.Linear(config.hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x)


def checkpoint_save(
    path: str | Path,
    model: ResidualMLP,
    config: ModelConfig,
    preprocessor: Any,
    class_names: list[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Derive input_dim from the input projection weight shape
    input_dim = model.input_proj.weight.shape[1]
    n_classes = model.head.weight.shape[0]
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "preprocessor": preprocessor,
            "class_names": class_names,
            "input_dim": input_dim,
            "n_classes": n_classes,
        },
        path,
    )


def checkpoint_load(path: str | Path, device: str = "cpu") -> tuple[ResidualMLP, ModelConfig, Any, list[str]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ModelConfig(**ckpt["config"])
    model = ResidualMLP(
        input_dim=ckpt["input_dim"],
        n_classes=ckpt["n_classes"],
        config=config,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, config, ckpt["preprocessor"], ckpt["class_names"]
