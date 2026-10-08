"""
Evaluación y visualización del TabTransformer entrenado.
 
Genera 4 gráficos PNG:
  1. confusion_matrix.png       — matriz de confusión para atomtype
  2. charge_scatter.png         — predicho vs real para charge
  3. charge_error_dist.png      — distribución del error de charge por clase
  4. per_class_accuracy.png     — accuracy por clase de atomtype
 
Uso:
    python evaluate_tabtransformer.py \
        --csv /ruta/aa_clean.csv \
        --checkpoint tabtransformer_best.pt \
        --out_dir ./resultados
"""
 
import argparse
import os
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error)
from sklearn.preprocessing import LabelEncoder, StandardScaler
 
# ── Columnas (deben coincidir con el training) ──────────────────────────────
NUMERIC_COLS = [
    "mass", "coordination", "avg_bond_len_A", "avg_angle_deg", "is_planar",
    "C_count", "H_count", "O_count", "N_count", "n_electroneg_neighbors",
    "n_electroneg_neighbors2", "bonds_double", "bonds_aromatic", "is_in_ring",
    "ring_size", "is_aromatic", "formal_charge", "gasteiger_charge",
]
ELEMENT_COLS  = ["element_C", "element_Cl", "element_F", "element_H",
                 "element_N", "element_O", "element_S"]
NEIGHBOR_COLS  = [f"neighbor_config_h{i}"  for i in range(32)]
NEIGHBOR2_COLS = [f"neighbor2_config_h{i}" for i in range(32)]
ALL_FEATURE_COLS = NUMERIC_COLS + ELEMENT_COLS + NEIGHBOR_COLS + NEIGHBOR2_COLS
 
 
# ── Modelo (copia exacta de TabTransformer.py) ──────────────────────────────
class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_model):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_features)])
 
    def forward(self, x):
        return torch.stack([p(x[:, i:i+1]) for i, p in enumerate(self.projections)], dim=1)
 
 
class TabTransformerModel(nn.Module):
    def __init__(self, n_features, n_classes, d_model=64, nhead=8,
                 num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        flat = n_features * d_model
        self.head_cat = nn.Sequential(nn.Linear(flat,128), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(128, n_classes))
        self.head_reg = nn.Sequential(nn.Linear(flat,128), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(128, 1))
 
    def forward(self, x):
        enc  = self.transformer(self.tokenizer(x))
        flat = enc.reshape(enc.size(0), -1)
        return self.head_cat(flat), self.head_reg(flat).squeeze(-1)
 
 
# ── Carga de datos ──────────────────────────────────────────────────────────
def load_data(csv_path, ckpt):
    df = pd.read_csv(csv_path)
    X  = df[ALL_FEATURE_COLS].values.astype(np.float32)
 
    # Reconstruir scaler desde el checkpoint
    scaler       = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_= ckpt["scaler_scale"]
    scaler.var_  = scaler.scale_ ** 2
    scaler.n_features_in_ = X.shape[1]
    X_scaled = scaler.transform(X)
 
    # Reconstruir label encoder desde el checkpoint
    le         = LabelEncoder()
    le.classes_= ckpt["label_encoder_classes"]
    y_cat      = le.transform(df["atomtype"].astype(str))
    y_reg      = df["charge"].values.astype(np.float32)
 
    return X_scaled, y_cat, y_reg, le
 
 
# ── Inferencia ──────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, X, batch_size=256, device="cpu"):
    model.eval()
    all_logits, all_charge = [], []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
        logits, charge = model(xb)
        all_logits.append(logits.cpu().numpy())
        all_charge.append(charge.cpu().numpy())
    logits = np.concatenate(all_logits)
    charge = np.concatenate(all_charge)
    return logits.argmax(axis=1), charge
 
 
# ── Gráficos ────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, classes, out_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
 
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Proporción")
 
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusión — atomtype")
 
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            if val > 0:
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=8, color=color)
 
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Guardado: {out_path}")
 
 
def plot_charge_scatter(y_true, y_pred, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="steelblue")
    lo = min(y_true.min(), y_pred.min()) - 0.05
    hi = max(y_true.max(), y_pred.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Ideal")
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.set_xlabel("Charge real"); ax.set_ylabel("Charge predicho")
    ax.set_title(f"Charge — predicho vs real\nMAE={mae:.4f}  RMSE={rmse:.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Guardado: {out_path}")
 
 
def plot_error_by_class(y_true_reg, y_pred_reg, y_true_cat, classes, out_path):
    errors = np.abs(y_true_reg - y_pred_reg)
    class_errors = {cls: [] for cls in classes}
    for e, c in zip(errors, y_true_cat):
        class_errors[classes[c]].append(e)
 
    labels  = [k for k in class_errors if class_errors[k]]
    medians = [np.median(class_errors[k]) for k in labels]
    order   = np.argsort(medians)[::-1]
 
    fig, ax = plt.subplots(figsize=(10, 5))
    data_sorted = [class_errors[labels[i]] for i in order]
    bp = ax.boxplot(data_sorted, patch_artist=True, medianprops={"color":"red","linewidth":2})
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
    ax.set_xticks(range(1, len(order)+1))
    ax.set_xticklabels([labels[i] for i in order], rotation=45, ha="right")
    ax.set_ylabel("|Error| en charge")
    ax.set_title("Distribución del error de charge por clase de atomtype")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Guardado: {out_path}")
 
 
def plot_per_class_accuracy(y_true, y_pred, classes, out_path):
    accs = []
    for i, cls in enumerate(classes):
        mask = y_true == i
        if mask.sum() == 0:
            accs.append(0.0)
        else:
            accs.append((y_pred[mask] == i).mean())
 
    counts = [int((y_true == i).sum()) for i in range(len(classes))]
    order  = np.argsort(accs)[::-1]
 
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(classes)), [accs[i] for i in order],
                  color="steelblue", edgecolor="white")
    for bar, idx in zip(bars, order):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"n={counts[idx]}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([classes[i] for i in order], rotation=45, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy por clase — atomtype")
    ax.axhline(y=np.mean(accs), color="red", linestyle="--",
               linewidth=1.2, label=f"Media: {np.mean(accs):.2f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Guardado: {out_path}")
 
 
# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        required=True,  help="CSV de datos")
    parser.add_argument("--checkpoint", default="tabtransformer_best.pt")
    parser.add_argument("--out_dir",    default="resultados")
    parser.add_argument("--d_model",    type=int, default=64)
    parser.add_argument("--nhead",      type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout",    type=float, default=0.1)
    args = parser.parse_args()
 
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # Cargar checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = list(ckpt["label_encoder_classes"])
    n_classes = len(classes)
    print(f"Checkpoint: epoch {ckpt['epoch']} | clases: {classes}")
 
    # Datos
    X, y_cat, y_reg, le = load_data(args.csv, ckpt)
 
    # Modelo
    model = TabTransformerModel(
        n_features=X.shape[1], n_classes=n_classes,
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 4,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
 
    # Predicciones sobre todo el dataset
    y_pred_cat, y_pred_reg = predict(model, X, device=device)
 
    # ── Métricas en consola ─────────────────────────────────────────────────
    acc  = (y_pred_cat == y_cat).mean()
    mae  = mean_absolute_error(y_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_reg, y_pred_reg))
 
    print(f"\n── Métricas globales ──")
    print(f"  Atomtype accuracy : {acc*100:.2f}%")
    print(f"  Charge MAE        : {mae:.4f}")
    print(f"  Charge RMSE       : {rmse:.4f}")
 
    print(f"\n── Reporte por clase (atomtype) ──")
    print(classification_report(y_cat, y_pred_cat, target_names=classes, zero_division=0))
 
    # ── Gráficos ────────────────────────────────────────────────────────────
    print(f"\nGenerando gráficos en '{args.out_dir}/'...")
    plot_confusion_matrix(y_cat, y_pred_cat, classes,
                          os.path.join(args.out_dir, "confusion_matrix.png"))
    plot_charge_scatter(y_reg, y_pred_reg,
                        os.path.join(args.out_dir, "charge_scatter.png"))
    plot_error_by_class(y_reg, y_pred_reg, y_cat, classes,
                        os.path.join(args.out_dir, "charge_error_by_class.png"))
    plot_per_class_accuracy(y_cat, y_pred_cat, classes,
                            os.path.join(args.out_dir, "per_class_accuracy.png"))
 
    print(f"\nListo. 4 PNG guardados en '{args.out_dir}/'")
 
 
if __name__ == "__main__":
    main()
