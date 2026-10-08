"""
TabTransformer for multi-task prediction:
  - atomtype: categorical classification
  - charge: continuous regression
 
Usage:
    python tabtransformer_atomtype_charge.py --csv path/to/data.csv
    python tabtransformer_atomtype_charge.py --csv path/to/data.csv --epochs 50 --batch_size 64
 
Expected CSV columns:
    mass, coordination, avg_bond_len_A, avg_angle_deg, is_planar,
    C_count, H_count, O_count, N_count, n_electroneg_neighbors,
    n_electroneg_neighbors2, bonds_double, bonds_aromatic, is_in_ring,
    ring_size, is_aromatic, formal_charge, gasteiger_charge,
    atomtype, charge,
    element_C, element_Cl, element_F, element_H, element_N, element_O, element_S,
    neighbor_config_h0..h31 (32 cols),
    neighbor2_config_h0..h31 (32 cols)
"""
 
import argparse
import math
import os
 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
 
# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
NUMERIC_COLS = [
    "mass", "coordination", "avg_bond_len_A", "avg_angle_deg", "is_planar",
    "C_count", "H_count", "O_count", "N_count", "n_electroneg_neighbors",
    "n_electroneg_neighbors2", "bonds_double", "bonds_aromatic", "is_in_ring",
    "ring_size", "is_aromatic", "formal_charge", "gasteiger_charge",
]
 
ELEMENT_COLS = [
    "element_C", "element_Cl", "element_F", "element_H",
    "element_N", "element_O", "element_S",
]
 
NEIGHBOR_COLS  = [f"neighbor_config_h{i}"  for i in range(32)]
NEIGHBOR2_COLS = [f"neighbor2_config_h{i}" for i in range(32)]
 
ALL_FEATURE_COLS = NUMERIC_COLS + ELEMENT_COLS + NEIGHBOR_COLS + NEIGHBOR2_COLS  # 89 total
 
TARGET_CAT  = "atomtype"
TARGET_REG  = "charge"
 
 
# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class MoleculeDataset(Dataset):
    def __init__(self, X: np.ndarray, y_cat: np.ndarray, y_reg: np.ndarray):
        self.X     = torch.tensor(X,     dtype=torch.float32)
        self.y_cat = torch.tensor(y_cat, dtype=torch.long)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        return self.X[idx], self.y_cat[idx], self.y_reg[idx]
 
 
# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class FeatureTokenizer(nn.Module):
    """Projects each scalar feature to d_model via a shared linear layer."""
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        # One linear projection per feature: (1,) → (d_model,)
        self.projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_features)
        ])
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features)  →  (B, n_features, d_model)
        tokens = [proj(x[:, i:i+1]) for i, proj in enumerate(self.projections)]
        return torch.stack(tokens, dim=1)
 
 
class TabTransformerModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
 
        flat_dim = n_features * d_model
 
        # Classification head (atomtype)
        self.head_cat = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )
 
        # Regression head (charge)
        self.head_reg = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
 
    def forward(self, x: torch.Tensor):
        tokens = self.tokenizer(x)                 # (B, n_features, d_model)
        encoded = self.transformer(tokens)          # (B, n_features, d_model)
        flat = encoded.reshape(encoded.size(0), -1) # (B, n_features * d_model)
 
        logits = self.head_cat(flat)               # (B, n_classes)
        charge = self.head_reg(flat).squeeze(-1)   # (B,)
        return logits, charge
 
 
# ─────────────────────────────────────────────
# Data loading & preprocessing
# ─────────────────────────────────────────────
def load_and_preprocess(csv_path: str):
    df = pd.read_csv(csv_path)
 
    # Ensure all expected feature columns exist
    missing = [c for c in ALL_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
 
    X = df[ALL_FEATURE_COLS].values.astype(np.float32)
 
    # Scale all features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
 
    # Encode atomtype labels
    le = LabelEncoder()
    y_cat = le.fit_transform(df[TARGET_CAT].astype(str)).astype(np.int64)
 
    # Regression target
    y_reg = df[TARGET_REG].values.astype(np.float32)
 
    return X, y_cat, y_reg, scaler, le
 
 
# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, loss_cat, loss_reg, device, alpha=1.0, beta=1.0):
    model.train()
    total_loss = 0.0
    for X, y_c, y_r in loader:
        X, y_c, y_r = X.to(device), y_c.to(device), y_r.to(device)
        optimizer.zero_grad()
        logits, charge_pred = model(X)
        l_cat = loss_cat(logits, y_c)
        l_reg = loss_reg(charge_pred, y_r)
        loss = alpha * l_cat + beta * l_reg
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)
 
 
@torch.no_grad()
def evaluate(model, loader, loss_cat, loss_reg, device, alpha=1.0, beta=1.0):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_charge_pred = []
    all_charge_true = []
 
    for X, y_c, y_r in loader:
        X, y_c, y_r = X.to(device), y_c.to(device), y_r.to(device)
        logits, charge_pred = model(X)
        l_cat = loss_cat(logits, y_c)
        l_reg = loss_reg(charge_pred, y_r)
        total_loss += (alpha * l_cat + beta * l_reg).item() * len(X)
 
        preds = logits.argmax(dim=1)
        correct += (preds == y_c).sum().item()
        total += len(y_c)
 
        all_charge_pred.append(charge_pred.cpu().numpy())
        all_charge_true.append(y_r.cpu().numpy())
 
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
 
    cp = np.concatenate(all_charge_pred)
    ct = np.concatenate(all_charge_true)
    mae  = np.mean(np.abs(cp - ct))
    rmse = np.sqrt(np.mean((cp - ct) ** 2))
 
    return avg_loss, accuracy, mae, rmse
 
 
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TabTransformer: atomtype + charge")
    parser.add_argument("--csv",        type=str,   required=True,  help="Path to input CSV")
    parser.add_argument("--epochs",     type=int,   default=30,     help="Training epochs")
    parser.add_argument("--batch_size", type=int,   default=128,    help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-3,   help="Learning rate")
    parser.add_argument("--d_model",    type=int,   default=64,     help="Token embedding dim")
    parser.add_argument("--nhead",      type=int,   default=8,      help="Attention heads")
    parser.add_argument("--num_layers", type=int,   default=3,      help="Transformer layers")
    parser.add_argument("--dropout",    type=float, default=0.1,    help="Dropout rate")
    parser.add_argument("--alpha",      type=float, default=1.0,    help="Weight for classification loss")
    parser.add_argument("--beta",       type=float, default=1.0,    help="Weight for regression loss")
    parser.add_argument("--val_split",  type=float, default=0.2,    help="Validation fraction")
    parser.add_argument("--seed",       type=int,   default=42,     help="Random seed")
    parser.add_argument("--save_model", type=str,   default="tabtransformer_best.pt",
                        help="Path to save best model checkpoint")
    args = parser.parse_args()
 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
 
    # ── Load data ──────────────────────────────
    print(f"Loading data from: {args.csv}")
    X, y_cat, y_reg, scaler, le = load_and_preprocess(args.csv)
    n_classes = len(le.classes_)
    print(f"  Samples:   {len(X)}")
    print(f"  Features:  {X.shape[1]}")
    print(f"  Atomtypes: {n_classes} → {list(le.classes_)}")
 
    # Use stratify only if all classes have at least 2 samples
    class_counts = np.bincount(y_cat)
    can_stratify = np.all(class_counts >= 2)
    if not can_stratify:
        rare = [le.classes_[i] for i, c in enumerate(class_counts) if c < 2]
        print(f"  Warning: classes with <2 samples (stratify disabled): {rare}")
 
    X_train, X_val, yc_train, yc_val, yr_train, yr_val = train_test_split(
        X, y_cat, y_reg,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=y_cat if can_stratify else None,
    )
 
    train_ds = MoleculeDataset(X_train, yc_train, yr_train)
    val_ds   = MoleculeDataset(X_val,   yc_val,   yr_val)
 
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
 
    # ── Model ──────────────────────────────────
    model = TabTransformerModel(
        n_features=X.shape[1],
        n_classes=n_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 4,
        dropout=args.dropout,
    ).to(device)
 
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
 
    # Compute class weights to handle imbalance
    class_counts = np.bincount(yc_train, minlength=n_classes).astype(float)
    class_weights = torch.tensor(1.0 / (class_counts + 1e-8), dtype=torch.float32).to(device)
    class_weights = class_weights / class_weights.sum() * n_classes
 
    loss_cat = nn.CrossEntropyLoss(weight=class_weights)
    loss_reg = nn.MSELoss()
 
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
 
    # ── Training ───────────────────────────────
    best_val_loss = float("inf")
    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Acc %':>8} {'MAE':>8} {'RMSE':>8}")
    print("-" * 62)
 
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_cat, loss_reg, device,
            alpha=args.alpha, beta=args.beta,
        )
        val_loss, acc, mae, rmse = evaluate(
            model, val_loader, loss_cat, loss_reg, device,
            alpha=args.alpha, beta=args.beta,
        )
        scheduler.step()
 
        print(f"{epoch:>6}  {train_loss:>12.4f}  {val_loss:>10.4f}  {acc*100:>7.2f}%  {mae:>8.4f}  {rmse:>8.4f}")
 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "label_encoder_classes": le.classes_,
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "args": vars(args),
            }, args.save_model)
 
    print(f"\nBest val loss: {best_val_loss:.4f} — model saved to {args.save_model}")
 
    # ── Final evaluation with best model ───────
    ckpt = torch.load(args.save_model, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, acc, mae, rmse = evaluate(model, val_loader, loss_cat, loss_reg, device)
    print("\n── Best model — Validation metrics ──")
    print(f"  Atomtype accuracy : {acc*100:.2f}%")
    print(f"  Charge MAE        : {mae:.4f}")
    print(f"  Charge RMSE       : {rmse:.4f}")
 
 
if __name__ == "__main__":
    main()
