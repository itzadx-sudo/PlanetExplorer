from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.mlp import ResidualMLP, ModelConfig, checkpoint_load
from model.preprocessor import TabularPreprocessor  # noqa: F401 — must be imported before torch.load unpickles it

# Checkpoints saved by train.py store TabularPreprocessor as __main__.TabularPreprocessor.
# Register it under that name so torch.load can find it regardless of how this module is invoked.
import __main__
if not hasattr(__main__, "TabularPreprocessor"):
    __main__.TabularPreprocessor = TabularPreprocessor


def _detect_device() -> str:
    try:
        import mlx.core  # noqa: F401
        return "mlx"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _bucket(prob: float, margin: float) -> str:
    if prob >= 0.90 and margin >= 0.30:
        return "High"
    if prob >= 0.75 and margin >= 0.15:
        return "Medium"
    return "Low"


def load_model(checkpoint_path: str | Path, device: str = "auto") -> tuple[Any, Any, list[str], str]:
    """Load checkpoint and return (model, preprocessor, class_names, resolved_device)."""
    checkpoint_path = Path(checkpoint_path)
    resolved = _detect_device() if device == "auto" else device

    if resolved == "mlx":
        model, config, preprocessor, class_names = _load_mlx(checkpoint_path)
    else:
        model, config, preprocessor, class_names = checkpoint_load(checkpoint_path, device=resolved)

    return model, preprocessor, class_names, resolved


def _load_mlx(checkpoint_path: Path):
    import mlx.core as mx
    import mlx.nn as mlx_nn
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ModelConfig(**ckpt["config"])

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

    mlx_model = MLXResidualMLP(ckpt["input_dim"], ckpt["n_classes"], config)

    # Copy weights from PyTorch checkpoint into MLX model
    sd = ckpt["state_dict"]

    def t(key: str):
        return mx.array(sd[key].numpy())

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

    return mlx_model, config, ckpt["preprocessor"], ckpt["class_names"]


def predict(
    model: Any,
    preprocessor: Any,
    df: pd.DataFrame,
    class_names: list[str],
    device: str = "cpu",
) -> pd.DataFrame:
    """Run inference and return a DataFrame with prediction columns appended."""
    X = preprocessor.transform_X(df).astype(np.float32)

    if device == "mlx":
        import mlx.core as mx
        import mlx.nn as mlx_nn

        x_mx = mx.array(X)
        logits = model(x_mx)
        probs = np.array(mlx_nn.softmax(logits, axis=-1).tolist())
    else:
        with torch.no_grad():
            logits = model(torch.from_numpy(X).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()

    pred_idx = probs.argmax(axis=1)
    pred_conf = probs.max(axis=1)
    top2_idx = np.argsort(probs, axis=1)[:, -2]
    top2_prob = probs[np.arange(len(probs)), top2_idx]
    margin = pred_conf - top2_prob

    return pd.DataFrame(
        {
            "prediction": [class_names[i] for i in pred_idx],
            "confidence": pred_conf.tolist(),
            "margin": margin.tolist(),
            "confidence_level": [_bucket(float(p), float(m)) for p, m in zip(pred_conf, margin)],
        }
    )


def predict_csv(
    checkpoint_path: str | Path,
    input_csv: str | Path,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Load a CSV, run inference, optionally write results, and return the results DataFrame."""
    model, preprocessor, class_names, device = load_model(checkpoint_path)
    df = pd.read_csv(input_csv)
    results = predict(model, preprocessor, df, class_names, device=device)
    if output_csv is not None:
        results.to_csv(output_csv, index=False)
    return results
