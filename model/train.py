import argparse
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.mlp import ResidualMLP, ModelConfig, checkpoint_save
from model.preprocessor import TabularPreprocessor

def get_device(requested_device: str):
    if requested_device == "auto":
        try:
            import mlx.core
            return "mlx"
        except ImportError:
            pass
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return requested_device

def train_pytorch(model, X_train, y_train, X_val, y_val, config, class_weights, device):
    print(f"Training on {device.upper()} using PyTorch...")
    model.to(device)
    
    if hasattr(torch, "compile") and device == "cuda":
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Could not compile model: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    val_X_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_y_t = torch.tensor(y_val, dtype=torch.long).to(device)
    
    best_val_f1 = -1
    patience_counter = 0
    best_state = None
    
    for epoch in range(config.max_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        with torch.no_grad():
            val_out = model(val_X_t)
            val_loss = criterion(val_out, val_y_t).item()
            val_preds = val_out.argmax(dim=1).cpu().numpy()
            
        val_acc = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average='macro')
        
        print(f"Epoch {epoch+1}/{config.max_epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Device: {device.upper()}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    model.load_state_dict(best_state)
    return model

def train_mlx(pytorch_model, X_train, y_train, X_val, y_val, config, class_weights):
    print("Training on Apple MLX...")
    import mlx.core as mx
    import mlx.nn as mlx_nn
    import mlx.optimizers as optim
    
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

    input_dim = pytorch_model.input_proj.weight.shape[1]
    n_classes = pytorch_model.head.weight.shape[0]
    model = MLXResidualMLP(input_dim, n_classes, config)
    
    # Initialize parameters
    mx.eval(model.parameters())
    
    optimizer = optim.AdamW(learning_rate=config.lr, weight_decay=config.weight_decay)
    
    # Class weights for loss
    mx_weights = mx.array(class_weights, dtype=mx.float32)
    
    def loss_fn(model, x, y):
        logits = model(x)
        # Weighted cross entropy
        loss = mlx_nn.losses.cross_entropy(logits, y)
        w = mx_weights[y]
        return mx.mean(loss * w)
        
    loss_and_grad_fn = mlx_nn.value_and_grad(model, loss_fn)
    
    X_train_mx = mx.array(X_train, dtype=mx.float32)
    y_train_mx = mx.array(y_train, dtype=mx.int32)
    X_val_mx = mx.array(X_val, dtype=mx.float32)
    y_val_mx = mx.array(y_val, dtype=mx.int32)
    
    best_val_f1 = -1
    patience_counter = 0
    best_state = None
    
    num_batches = int(np.ceil(len(X_train) / config.batch_size))
    
    for epoch in range(config.max_epochs):
        model.train()
        indices = np.random.permutation(len(X_train))
        epoch_loss = 0
        
        for i in range(num_batches):
            batch_idx = indices[i*config.batch_size : (i+1)*config.batch_size]
            batch_x = mx.take(X_train_mx, mx.array(batch_idx), axis=0)
            batch_y = mx.take(y_train_mx, mx.array(batch_idx), axis=0)
            
            loss, grads = loss_and_grad_fn(model, batch_x, batch_y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            epoch_loss += loss.item()
            
        model.eval()
        val_logits = model(X_val_mx)
        val_preds = np.argmax(np.array(val_logits), axis=1)
        
        val_acc = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average='macro')
        
        print(f"Epoch {epoch+1}/{config.max_epochs} | Loss: {epoch_loss/num_batches:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Device: MLX")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            import mlx.utils
            best_state = mlx.utils.tree_map(lambda x: mx.array(x), model.parameters())
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    # Restore best state
    model.update(best_state)
    
    # Translate MLX model back to PyTorch state_dict so we can use checkpoint_save
    sd = pytorch_model.state_dict()
    
    def pt(mx_arr):
        return torch.from_numpy(np.array(mx_arr))
        
    sd["input_proj.weight"] = pt(best_state["input_proj"]["weight"])
    if "bias" in best_state["input_proj"]:
        sd["input_proj.bias"] = pt(best_state["input_proj"]["bias"])
        
    for i in range(config.num_blocks):
        sd[f"blocks.{i}.block.0.weight"] = pt(best_state["blocks"][i]["norm"]["weight"])
        sd[f"blocks.{i}.block.0.bias"] = pt(best_state["blocks"][i]["norm"]["bias"])
        sd[f"blocks.{i}.block.1.weight"] = pt(best_state["blocks"][i]["fc1"]["weight"])
        sd[f"blocks.{i}.block.1.bias"] = pt(best_state["blocks"][i]["fc1"]["bias"])
        sd[f"blocks.{i}.block.4.weight"] = pt(best_state["blocks"][i]["fc2"]["weight"])
        sd[f"blocks.{i}.block.4.bias"] = pt(best_state["blocks"][i]["fc2"]["bias"])
        
    sd["head.weight"] = pt(best_state["head"]["weight"])
    if "bias" in best_state["head"]:
        sd["head.bias"] = pt(best_state["head"]["bias"])
        
    pytorch_model.load_state_dict(sd)
    return pytorch_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="data/processed/train.csv")
    parser.add_argument("--device", choices=["auto", "mlx", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    
    config = ModelConfig()
    if args.epochs:
        config.max_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
        
    device = get_device(args.device)
    
    print(f"Loading data from {args.train_file}...")
    df = pd.read_csv(args.train_file)
    
    # 85/15 stratified split
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["koi_disposition"], random_state=42)
    
    class_names = sorted(df["koi_disposition"].unique().tolist())
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    y_train = train_df["koi_disposition"].map(class_to_idx).values
    y_val = val_df["koi_disposition"].map(class_to_idx).values
    
    # Compute class weights
    class_counts = np.bincount(y_train, minlength=len(class_names))
    total_samples = len(y_train)
    class_weights = total_samples / (len(class_names) * class_counts)
    
    preprocessor = TabularPreprocessor()
    preprocessor.fit(train_df, target_col="koi_disposition", drop_cols=["kepid"])
    
    X_train = preprocessor.transform_X(train_df).astype(np.float32)
    X_val = preprocessor.transform_X(val_df).astype(np.float32)
    
    # PyTorch Model (used natively or as a shell for saving)
    pytorch_model = ResidualMLP(input_dim=X_train.shape[1], n_classes=len(class_names), config=config)
    
    if device == "mlx":
        pytorch_model = train_mlx(pytorch_model, X_train, y_train, X_val, y_val, config, class_weights)
    else:
        pytorch_model = train_pytorch(pytorch_model, X_train, y_train, X_val, y_val, config, class_weights, device)
        
    # Save checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"artifacts/mlp_{timestamp}_min.pt"
    print(f"Saving checkpoint to {checkpoint_path}...")
    checkpoint_save(checkpoint_path, pytorch_model, config, preprocessor, class_names)
    print("Done!")

if __name__ == "__main__":
    main()
