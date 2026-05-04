from __future__ import annotations

import io
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.mlp import ResidualMLP, ModelConfig, checkpoint_load
from model.preprocessor import TabularPreprocessor
from model.device import get_device
from model.mlx_model import build_mlx_model, copy_pytorch_weights_to_mlx


class _CompatUnpickler(pickle.Unpickler):
    """Remaps __main__.TabularPreprocessor to its canonical module path.

    Checkpoints trained via train.py pickle TabularPreprocessor under __main__
    because it was defined at module level there. This lets old checkpoints load
    without namespace pollution.
    """
    def find_class(self, module, name):
        if module == "__main__" and name == "TabularPreprocessor":
            return TabularPreprocessor
        return super().find_class(module, name)


class _CompatPickleModule:
    """Minimal shim so torch.load accepts our custom unpickler."""
    Unpickler = _CompatUnpickler

    @staticmethod
    def dump(obj, f):
        pickle.dump(obj, f)

    @staticmethod
    def dumps(obj):
        return pickle.dumps(obj)


def _torch_load_compat(path: Path, map_location):
    return torch.load(path, map_location=map_location, pickle_module=_CompatPickleModule, weights_only=False)


def _bucket(prob: float, margin: float) -> str:
    if prob >= 0.90 and margin >= 0.30:
        return "High"
    if prob >= 0.75 and margin >= 0.15:
        return "Medium"
    return "Low"


def load_model(checkpoint_path: str | Path, device: str = "auto") -> tuple[Any, Any, list[str], str]:
    checkpoint_path = Path(checkpoint_path)
    resolved = get_device(device)

    if resolved == "mlx":
        model, config, preprocessor, class_names = _load_mlx(checkpoint_path)
    else:
        model, config, preprocessor, class_names = _load_pytorch(checkpoint_path, resolved)

    return model, preprocessor, class_names, resolved


def _load_pytorch(checkpoint_path: Path, device: str):
    ckpt = _torch_load_compat(checkpoint_path, map_location=device)
    config = ModelConfig(**ckpt["config"])
    model = ResidualMLP(input_dim=ckpt["input_dim"], n_classes=ckpt["n_classes"], config=config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, config, ckpt["preprocessor"], ckpt["class_names"]


def _load_mlx(checkpoint_path: Path):
    import mlx.core as mx

    ckpt = _torch_load_compat(checkpoint_path, map_location="cpu")
    config = ModelConfig(**ckpt["config"])
    mlx_model = build_mlx_model(ckpt["input_dim"], ckpt["n_classes"], config)
    copy_pytorch_weights_to_mlx(mlx_model, ckpt["state_dict"], config)
    mx.eval(mlx_model.parameters())
    return mlx_model, config, ckpt["preprocessor"], ckpt["class_names"]


def predict(
    model: Any,
    preprocessor: Any,
    df: pd.DataFrame,
    class_names: list[str],
    device: str = "cpu",
) -> pd.DataFrame:
    X = preprocessor.transform_X(df).astype(np.float32)

    if device == "mlx":
        import mlx.core as mx
        import mlx.nn as mlx_nn

        logits = model(mx.array(X))
        probs = np.array(mlx_nn.softmax(logits, axis=-1).tolist())
    else:
        with torch.no_grad():
            logits = model(torch.from_numpy(X).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()

    pred_idx = probs.argmax(axis=1)
    pred_conf = probs.max(axis=1)
    top2_idx = np.argsort(probs, axis=1)[:, -2]
    margin = pred_conf - probs[np.arange(len(probs)), top2_idx]

    return pd.DataFrame({
        "prediction": [class_names[i] for i in pred_idx],
        "confidence": pred_conf.tolist(),
        "margin": margin.tolist(),
        "confidence_level": [_bucket(float(p), float(m)) for p, m in zip(pred_conf, margin)],
    })


def predict_csv(
    checkpoint_path: str | Path,
    input_csv: str | Path,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    model, preprocessor, class_names, device = load_model(checkpoint_path)
    df = pd.read_csv(input_csv)
    results = predict(model, preprocessor, df, class_names, device=device)
    if output_csv is not None:
        results.to_csv(output_csv, index=False)
    return results
