"""
validate_external.py
--------------------
Valida los modelos sobre moléculas externas no vistas durante el entrenamiento.

Modelos utilizados:
  - AA clasificación (atomtype): Random Forest E3 (clf_AA.joblib)
  - AA regresión (charge):       TabTransformer  (reg_AA.pt)
  - UA clasificación (atomtype): Random Forest E3 (clf_UA.joblib)
  - UA regresión (charge):       Random Forest E3 (reg_UA.joblib)

Uso:
    python validate_external.py \
        --molecules carpeta_con_mol2_e_itp/ \
        --models_AA results_AA/ \
        --models_UA results_UA/ \
        --artifacts processed/preprocessing_artifacts.json \
        --pipeline_dir /home/marazi/proyectoParam/scripts/prueba2/version2(calude) \
        --output_dir validation_external/
"""

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from joblib import load
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"]  = 11

# =============================================================================
# CONFIG
# =============================================================================

TARGET_CLF   = "atomtype"
TARGET_REG   = "charge"
COLS_TO_DROP = ["degree_of_unsat", "n_dihedrals", "bonds_single", "tripos_type"]
BOOL_COLS    = ["is_planar", "is_in_ring", "is_aromatic"]
TOP_N        = 20

ATOMTYPE_REMAP = {
    "NOpt":  "N",
    "NPri":  "N",
    "CLAro": "CL",
    "SDmso": "S",
    "CAro":  "C",
}

# Features en el orden exacto que espera el TabTransformer
NUMERIC_COLS = [
    "mass", "coordination", "avg_bond_len_A", "avg_angle_deg", "is_planar",
    "C_count", "H_count", "O_count", "N_count", "n_electroneg_neighbors",
    "n_electroneg_neighbors2", "bonds_double", "bonds_aromatic", "is_in_ring",
    "ring_size", "is_aromatic", "formal_charge", "gasteiger_charge",
]
ELEMENT_COLS   = ["element_C", "element_Cl", "element_F", "element_H",
                  "element_N", "element_O", "element_S"]
NEIGHBOR_COLS  = [f"neighbor_config_h{i}"  for i in range(32)]
NEIGHBOR2_COLS = [f"neighbor2_config_h{i}" for i in range(32)]
TABTRANSFORMER_COLS = NUMERIC_COLS + ELEMENT_COLS + NEIGHBOR_COLS + NEIGHBOR2_COLS


# =============================================================================
# TABTRANSFORMER — arquitectura
# =============================================================================

class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_model):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_features)
        ])

    def forward(self, x):
        tokens = [proj(x[:, i:i+1]) for i, proj in enumerate(self.projections)]
        return torch.stack(tokens, dim=1)


class TabTransformerModel(nn.Module):
    def __init__(self, n_features, n_classes, d_model=64, nhead=8,
                 num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        flat_dim = n_features * d_model
        self.head_cat = nn.Sequential(
            nn.Linear(flat_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )
        self.head_reg = nn.Sequential(
            nn.Linear(flat_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        tokens  = self.tokenizer(x)
        encoded = self.transformer(tokens)
        flat    = encoded.reshape(encoded.size(0), -1)
        return self.head_cat(flat), self.head_reg(flat).squeeze(-1)


def load_tabtransformer(pt_path, device):
    ckpt       = torch.load(pt_path, map_location=device, weights_only=False)
    args       = ckpt["args"]
    n_classes  = len(ckpt["label_encoder_classes"])
    n_features = len(TABTRANSFORMER_COLS)
    model = TabTransformerModel(
        n_features=n_features, n_classes=n_classes,
        d_model=args.get("d_model", 64), nhead=args.get("nhead", 8),
        num_layers=args.get("num_layers", 3),
        dim_feedforward=args.get("d_model", 64) * 4,
        dropout=args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaler        = StandardScaler()
    scaler.mean_  = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.n_features_in_ = n_features
    return model, scaler


def predict_charge_tabtransformer(model, scaler, df_proc, device):
    df_proc = df_proc.copy()
    for col in TABTRANSFORMER_COLS:
        if col not in df_proc.columns:
            df_proc[col] = 0
    X = df_proc[TABTRANSFORMER_COLS].values.astype(np.float32)
    X = scaler.transform(X)
    with torch.no_grad():
        _, charge_pred = model(torch.tensor(X, dtype=torch.float32).to(device))
    return charge_pred.cpu().numpy()


# =============================================================================
# HELPERS
# =============================================================================

def save(fig, out_dir, name):
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


def section(title):
    print(f"\n{'='*55}\n  {title}\n{'='*55}")


# =============================================================================
# PARSERS
# =============================================================================

def parse_mol2(mol2_path):
    atoms, bonds = {}, []
    section_name = None
    with open(mol2_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("@<TRIPOS>ATOM"):
                section_name = "atom"; continue
            if line.startswith("@<TRIPOS>BOND"):
                section_name = "bond"; continue
            if line.startswith("@<TRIPOS>"):
                section_name = None; continue
            if section_name == "atom" and line.strip():
                parts = line.split()
                if len(parts) < 6:
                    continue
                atom_id     = int(parts[0])
                atom_name   = parts[1]
                x, y, z     = float(parts[2]), float(parts[3]), float(parts[4])
                tripos_type = parts[5]
                elem_raw    = tripos_type.split(".")[0]
                elem_raw    = ''.join(c for c in elem_raw if c.isalpha())
                ELEMENT_NORMALIZE = {
                    "HC": "H", "HO": "H", "HN": "H",
                    "CL": "Cl", "BR": "Br", "SI": "Si",
                    "FE": "Fe", "ZN": "Zn", "CA": "Ca",
                    "MG": "Mg", "NA": "Na",
                }
                element = ELEMENT_NORMALIZE.get(elem_raw.upper(),
                              elem_raw.capitalize() if len(elem_raw) > 1 else elem_raw)
                atoms[atom_id] = {
                    "atom_name":   atom_name,
                    "tripos_type": tripos_type,
                    "element":     element,
                    "xyz":         (x, y, z),
                }
            if section_name == "bond" and line.strip():
                parts = line.split()
                if len(parts) >= 4:
                    bonds.append((int(parts[1]), int(parts[2]), parts[3]))
    return atoms, bonds


def parse_itp(itp_path):
    by_index = {}
    in_block = False
    with open(itp_path) as f:
        for line in f:
            line = line.strip()
            if "[ atoms ]" in line:
                in_block = True; continue
            if in_block and line.startswith("["):
                break
            if not line or line.startswith(";"):
                continue
            if in_block:
                parts = line.split()
                if len(parts) >= 7:
                    atom_idx           = int(parts[0])
                    by_index[atom_idx] = {
                        "atomtype": parts[1],
                        "charge":   float(parts[6]),
                    }
    return by_index


# =============================================================================
# EXTRACCIÓN DE FEATURES
# =============================================================================

def extract_features_from_mol2(mol2_path, pipeline_dir):
    sys.path.append(str(pipeline_dir))
    from extract_dataset_v3 import (
        build_rdkit_mol, bond_length, angle_deg, is_planar_geometry,
        ATOMIC_MASSES, PLANAR_TYPES, ELECTRONEGATIVE,
    )

    mol_name     = mol2_path.stem
    atoms, bonds = parse_mol2(mol2_path)

    graph = {aid: [] for aid in atoms}
    for a1, a2, btype in bonds:
        if a1 in graph and a2 in graph:
            graph[a1].append((a2, btype))
            graph[a2].append((a1, btype))

    dihedral_counts = {aid: 0 for aid in atoms}
    for a1, a2, _ in bonds:
        if a1 not in graph or a2 not in graph:
            continue
        for (a, _) in graph[a1]:
            if a == a2: continue
            for (d, _) in graph[a2]:
                if d == a1: continue
                for idx in (a, a1, a2, d):
                    if idx in dihedral_counts:
                        dihedral_counts[idx] += 1

    mol, id_to_idx = build_rdkit_mol(atoms, bonds)
    has_rdkit      = mol is not None
    has_gasteiger  = False
    ring_info      = None

    if has_rdkit:
        try:
            from rdkit.Chem import rdPartialCharges
            rdPartialCharges.ComputeGasteigerCharges(mol)
            has_gasteiger = True
        except Exception:
            pass
        ring_info = mol.GetRingInfo()

    rows = []
    for atom_id in sorted(atoms):
        atom        = atoms[atom_id]
        tripos_type = atom["tripos_type"]
        atom_name   = atom["atom_name"]
        element     = atom["element"]
        xyz_i       = atom["xyz"]
        mass        = ATOMIC_MASSES.get(element, 0.0)
        neighbors   = graph[atom_id]
        coordination= len(neighbors)

        neighbor_info    = []
        bond_lengths     = []
        neighbor_xyzs    = []
        elem_counts      = {"C": 0, "H": 0, "O": 0, "N": 0}
        bond_type_counts = {"single": 0, "double": 0, "aromatic": 0}
        n_electroneg_1   = 0
        bonded_elements  = []

        for nb_id, btype_str in neighbors:
            nb      = atoms[nb_id]
            nb_elem = nb["element"]
            nb_xyz  = nb["xyz"]
            bonded_elements.append(nb_elem)
            bond_lengths.append(bond_length(xyz_i, nb_xyz))
            neighbor_xyzs.append(nb_xyz)
            if nb_elem in elem_counts:
                elem_counts[nb_elem] += 1
            if nb_elem in ELECTRONEGATIVE:
                n_electroneg_1 += 1
            btype_norm = btype_str.lower()
            if btype_norm == "ar":
                bond_type_counts["aromatic"] += 1
            elif btype_norm in ("2", "3"):
                bond_type_counts["double"] += 1
            else:
                bond_type_counts["single"] += 1
            neighbor_info.append(f"{nb_elem}{btype_str}")

        bonded_to_element = (bonded_elements[0] if coordination == 1 and bonded_elements
                             else ",".join(sorted(bonded_elements)) if coordination > 1 else "")
        neighbor_config  = "-".join(sorted(neighbor_info))
        avg_bond_len     = float(np.mean(bond_lengths)) if bond_lengths else 0.0

        seen2 = set()
        neighbors2_elems = []
        for nb_id, _ in neighbors:
            for nb2_id, _ in graph[nb_id]:
                if nb2_id != atom_id and nb2_id not in seen2:
                    seen2.add(nb2_id)
                    neighbors2_elems.append(atoms[nb2_id]["element"])
        neighbor2_config = "-".join(sorted(neighbors2_elems))
        n_electroneg_2   = sum(1 for e in neighbors2_elems if e in ELECTRONEGATIVE)

        nb_list = [nb_id for nb_id, _ in neighbors]
        angles  = [angle_deg(atoms[nb_list[j]]["xyz"], xyz_i, atoms[nb_list[k]]["xyz"])
                   for j in range(len(nb_list)) for k in range(j+1, len(nb_list))]
        avg_angle = float(np.mean(angles)) if angles else 0.0

        if tripos_type in PLANAR_TYPES:
            planar = True
        elif coordination >= 3:
            planar = is_planar_geometry(xyz_i, neighbor_xyzs)
        else:
            planar = False

        if has_rdkit and id_to_idx:
            rdkit_idx = id_to_idx.get(atom_id)
            if rdkit_idx is not None:
                rd_atom       = mol.GetAtomWithIdx(rdkit_idx)
                is_in_ring    = ring_info.NumAtomRings(rdkit_idx) > 0
                atom_rings    = [r for r in ring_info.AtomRings() if rdkit_idx in r]
                ring_size     = min(len(r) for r in atom_rings) if atom_rings else 0
                is_aromatic   = rd_atom.GetIsAromatic()
                formal_charge = rd_atom.GetFormalCharge()
                if has_gasteiger:
                    try:
                        g = float(rd_atom.GetDoubleProp("_GasteigerCharge"))
                        gasteiger_charge = 0.0 if (math.isnan(g) or math.isinf(g)) else g
                    except Exception:
                        gasteiger_charge = 0.0
                else:
                    gasteiger_charge = 0.0
            else:
                is_in_ring = is_aromatic = False
                ring_size = formal_charge = 0
                gasteiger_charge = 0.0
        else:
            is_in_ring    = tripos_type in {"C.ar", "N.ar"}
            ring_size     = 6 if is_in_ring else 0
            is_aromatic   = is_in_ring
            formal_charge = 0
            gasteiger_charge = 0.0

        degree_of_unsat = bond_type_counts["double"] + bond_type_counts["aromatic"]

        rows.append({
            "molecule":                mol_name,
            "atom_id":                 atom_id,
            "atom_name":               atom_name,
            "tripos_type":             tripos_type,
            "element":                 element,
            "mass":                    round(mass, 4),
            "coordination":            coordination,
            "avg_bond_len_A":          round(avg_bond_len, 4),
            "avg_angle_deg":           round(avg_angle, 4),
            "is_planar":               planar,
            "C_count":                 elem_counts["C"],
            "H_count":                 elem_counts["H"],
            "O_count":                 elem_counts["O"],
            "N_count":                 elem_counts["N"],
            "neighbor_config":         neighbor_config,
            "neighbor2_config":        neighbor2_config,
            "n_dihedrals":             dihedral_counts[atom_id],
            "n_electroneg_neighbors":  n_electroneg_1,
            "n_electroneg_neighbors2": n_electroneg_2,
            "bonds_single":            bond_type_counts["single"],
            "bonds_double":            bond_type_counts["double"],
            "bonds_aromatic":          bond_type_counts["aromatic"],
            "is_in_ring":              is_in_ring,
            "ring_size":               ring_size,
            "is_aromatic":             is_aromatic,
            "formal_charge":           formal_charge,
            "degree_of_unsat":         degree_of_unsat,
            "gasteiger_charge":        round(gasteiger_charge, 6),
            "bonded_to_element":       bonded_to_element,
        })

    return pd.DataFrame(rows)


# =============================================================================
# PREPROCESAMIENTO
# =============================================================================

def preprocess(df, artifacts, scheme):
    df = df.copy()
    for col in COLS_TO_DROP:
        if col in df.columns:
            df = df.drop(columns=[col])
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)
    ohe_cats = artifacts[scheme]["ohe_categories"]
    for col, cats in ohe_cats.items():
        if col not in df.columns:
            continue
        for cat in cats:
            df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
        df = df.drop(columns=[col])
    hash_cols = artifacts[scheme]["hash_cols"]
    for col, n_features in hash_cols.items():
        if col not in df.columns:
            continue
        hasher    = FeatureHasher(n_features=n_features, input_type="string")
        hashed    = hasher.transform(df[col].fillna("").apply(lambda x: [x])).toarray()
        col_names = [f"{col}_h{i}" for i in range(n_features)]
        hashed_df = pd.DataFrame(hashed, columns=col_names, index=df.index, dtype=int)
        df = pd.concat([df.drop(columns=[col]), hashed_df], axis=1)
    return df


# =============================================================================
# SPLIT AA / UA
# =============================================================================

def split_AA_UA(df):
    POLAR = {"N", "O"}

    def is_AA(row):
        elem, bonded = row["element"], row["bonded_to_element"]
        if elem not in ("C", "H"): return True
        if elem == "C": return True
        if elem == "H" and bonded in ("C", *POLAR): return True
        return False

    def is_UA(row):
        elem, bonded = row["element"], row["bonded_to_element"]
        if elem not in ("C", "H"): return True
        if elem == "C": return True
        if elem == "H" and bonded in POLAR: return True
        return False

    return df[df.apply(is_AA, axis=1)].copy(), df[df.apply(is_UA, axis=1)].copy()


# =============================================================================
# FIGURAS
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, title, out_dir, fname):
    classes = sorted(set(y_true) | set(y_pred))
    cm      = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(max(8, len(classes)), max(6, len(classes)-2)))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(title)
    plt.tight_layout()
    save(fig, out_dir, fname)


def plot_scatter(y_true, y_pred, mae, rmse, r2, title, out_dir, fname):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, color="#378ADD",
               edgecolors="white", linewidth=0.3, s=50)
    lims = [min(min(y_true), min(y_pred)) - 0.05,
            max(max(y_true), max(y_pred)) + 0.05]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Predicción perfecta")
    ax.set_xlabel("Charge real (e)")
    ax.set_ylabel("Charge predicha (e)")
    ax.set_title(f"{title}\nMAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")
    ax.legend()
    plt.tight_layout()
    save(fig, out_dir, fname)


def plot_residuals(y_true, y_pred, title, out_dir, fname):
    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.6, color="#534AB7",
                    edgecolors="white", linewidth=0.3, s=50)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Charge predicha (e)")
    axes[0].set_ylabel("Residuo (real - predicho)")
    axes[0].set_title("Residuos vs Charge predicha")
    axes[1].hist(residuals, bins=20, color="#534AB7", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].axvline(residuals.mean(), color="orange", linestyle="--",
                    label=f"Media: {residuals.mean():.4f}")
    axes[1].set_xlabel("Residuo (e)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribución de residuos")
    axes[1].legend()
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    save(fig, out_dir, fname)


def plot_error_by_atomtype(df_results, title, out_dir, fname):
    df_results = df_results.copy()
    df_results["abs_error"] = abs(df_results["charge_real"] - df_results["charge_pred"])
    grouped = df_results.groupby("atomtype_real")["abs_error"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#ef5350" if v > 0.1 else "#378ADD" for v in grouped.values]
    ax.bar(grouped.index, grouped.values, color=colors)
    ax.axhline(0.05, color="orange", linestyle="--", linewidth=1, label="Umbral 0.05 e")
    ax.set_xlabel("Atomtype real")
    ax.set_ylabel("MAE (e)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    save(fig, out_dir, fname)


def plot_atomtype_comparison(y_true, y_pred, title, out_dir, fname):
    df = pd.DataFrame({"real": y_true, "pred": y_pred})
    classes     = sorted(set(y_true) | set(y_pred))
    real_counts = df["real"].value_counts().reindex(classes, fill_value=0)
    pred_counts = df["pred"].value_counts().reindex(classes, fill_value=0)
    x = np.arange(len(classes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(classes)), 5))
    ax.bar(x - w/2, real_counts.values, w, label="Real",     color="#378ADD")
    ax.bar(x + w/2, pred_counts.values, w, label="Predicho", color="#D85A30")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Cantidad de átomos")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    save(fig, out_dir, fname)


# =============================================================================
# VALIDACIÓN PRINCIPAL
# =============================================================================

def validate(mol2_path, itp_path, clf, reg, artifacts, scheme,
             pipeline_dir, out_dir, label,
             tab_model=None, tab_scaler=None, device=None):

    section(f"Validación {label} — {scheme}")
    mol_name = mol2_path.stem

    df_raw   = extract_features_from_mol2(mol2_path, pipeline_dir)
    df_aa, df_ua = split_AA_UA(df_raw)
    df_scheme = df_aa if scheme == "AA" else df_ua

    itp_data = parse_itp(itp_path)
    df_scheme = df_scheme.copy()
    df_scheme["atomtype_real"] = df_scheme["atom_id"].map(
        lambda x: itp_data.get(x, {}).get("atomtype", None))
    df_scheme["charge_real"] = df_scheme["atom_id"].map(
        lambda x: itp_data.get(x, {}).get("charge", None))
    df_scheme["atomtype_real"] = df_scheme["atomtype_real"].replace(ATOMTYPE_REMAP)
    df_scheme = df_scheme.dropna(subset=["atomtype_real", "charge_real"])

    print(f"  Átomos para validación: {len(df_scheme)}")
    print(f"  Atomtypes reales únicos: {sorted(df_scheme['atomtype_real'].unique())}")

    df_proc = preprocess(df_scheme, artifacts, scheme)

    # Clasificación → siempre RF
    expected_cols = list(clf.feature_names_in_) if hasattr(clf, "feature_names_in_") else []
    if expected_cols:
        for col in expected_cols:
            if col not in df_proc.columns:
                df_proc[col] = 0
        df_proc_clf = df_proc[expected_cols]
    else:
        df_proc_clf = df_proc

    y_clf_pred = clf.predict(df_proc_clf)

    # Regresión → TabTransformer para AA, RF para UA
    if scheme == "AA" and tab_model is not None:
        y_reg_pred = predict_charge_tabtransformer(tab_model, tab_scaler, df_proc.copy(), device)
    else:
        y_reg_pred = reg.predict(df_proc_clf)

    y_clf_true = df_scheme["atomtype_real"].values
    y_reg_true = df_scheme["charge_real"].values.astype(float)

    # Métricas clasificación
    acc    = accuracy_score(y_clf_true, y_clf_pred)
    f1_mac = f1_score(y_clf_true, y_clf_pred, average="macro",    zero_division=0)
    f1_w   = f1_score(y_clf_true, y_clf_pred, average="weighted", zero_division=0)
    print(f"\n  Clasificación:")
    print(f"    Accuracy:    {acc:.4f}")
    print(f"    F1 macro:    {f1_mac:.4f}")
    print(f"    F1 weighted: {f1_w:.4f}")
    print()
    print(classification_report(y_clf_true, y_clf_pred, zero_division=0))

    # Métricas regresión
    mae  = mean_absolute_error(y_reg_true, y_reg_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_true, y_reg_pred))
    r2   = r2_score(y_reg_true, y_reg_pred)
    print(f"  Regresión ({'TabTransformer' if scheme == 'AA' else 'RF'}):")
    print(f"    MAE:  {mae:.4f} e")
    print(f"    RMSE: {rmse:.4f} e")
    print(f"    R²:   {r2:.4f}")

    df_results = df_scheme[["molecule", "atom_id", "atom_name",
                             "element", "atomtype_real", "charge_real"]].copy()
    df_results["atomtype_pred"]    = y_clf_pred
    df_results["charge_pred"]      = np.round(y_reg_pred, 4)
    df_results["clf_correct"]      = (df_results["atomtype_real"] == df_results["atomtype_pred"])
    df_results["charge_error"]     = np.round(y_reg_true - y_reg_pred, 4)
    df_results["charge_abs_error"] = np.abs(df_results["charge_error"]).round(4)

    csv_path = out_dir / f"results_{mol_name}_{scheme}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n  → {csv_path.name}")

    prefix = f"{mol_name}_{scheme}"
    plot_confusion_matrix(y_clf_true, y_clf_pred,
        f"Matriz de confusión — {mol_name} {scheme}\n(validación externa)",
        out_dir, f"fig_confusion_{prefix}.png")
    plot_scatter(y_reg_true, y_reg_pred, mae, rmse, r2,
        f"Charge real vs predicha — {mol_name} {scheme}",
        out_dir, f"fig_scatter_{prefix}.png")
    plot_residuals(y_reg_true, y_reg_pred,
        f"Análisis de residuos — {mol_name} {scheme}",
        out_dir, f"fig_residuals_{prefix}.png")
    plot_error_by_atomtype(df_results,
        f"MAE de charge por atomtype — {mol_name} {scheme}",
        out_dir, f"fig_error_by_atomtype_{prefix}.png")
    plot_atomtype_comparison(y_clf_true, y_clf_pred,
        f"Atomtypes reales vs predichos — {mol_name} {scheme}",
        out_dir, f"fig_atomtype_comparison_{prefix}.png")

    return {
        "molecule": mol_name,
        "scheme":   scheme,
        "n_atoms":  len(df_scheme),
        "reg_model": "TabTransformer" if scheme == "AA" else "RF_E3",
        "atomtypes_real": sorted(df_scheme["atomtype_real"].unique().tolist()),
        "classification": {
            "accuracy":    round(acc, 4),
            "f1_macro":    round(f1_mac, 4),
            "f1_weighted": round(f1_w, 4),
        },
        "regression": {
            "mae":  round(mae, 4),
            "rmse": round(rmse, 4),
            "r2":   round(r2, 4),
        }
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validación externa — RF E3 (clf) + TabTransformer (reg AA) + RF E3 (reg UA)"
    )
    parser.add_argument("--molecules",    required=True)
    parser.add_argument("--models_AA",    required=True,
                        help="Carpeta con clf_AA.joblib y reg_AA.pt")
    parser.add_argument("--models_UA",    required=True,
                        help="Carpeta con clf_UA.joblib y reg_UA.joblib")
    parser.add_argument("--artifacts",    required=True)
    parser.add_argument("--pipeline_dir", required=True)
    parser.add_argument("--output_dir",   default="validation_external/")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cargar modelos RF
    clf_aa = load(Path(args.models_AA) / "clf_AA.joblib")
    clf_ua = load(Path(args.models_UA) / "clf_UA.joblib")
    reg_ua = load(Path(args.models_UA) / "reg_UA.joblib")

    # Cargar TabTransformer para regresión AA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tab_model, tab_scaler = load_tabtransformer(
        Path(args.models_AA) / "reg_AA.pt", device)
    print("✔ TabTransformer cargado para regresión AA")

    with open(args.artifacts) as f:
        artifacts = json.load(f)

    pipeline_dir = Path(args.pipeline_dir)
    mol2_files   = sorted(Path(args.molecules).glob("*.mol2"))
    if not mol2_files:
        sys.exit("No se encontraron archivos .mol2")

    all_results = []

    for mol2_path in mol2_files:
        itp_candidates = list(Path(args.molecules).glob(f"{mol2_path.stem}*.itp"))
        if not itp_candidates:
            itp_candidates = list(Path(args.molecules).glob(f"{mol2_path.stem.upper()}*.itp"))
        if not itp_candidates:
            print(f"  [SKIP] Sin .itp para {mol2_path.name}")
            continue

        itp_path = itp_candidates[0]
        print(f"\n✔ Procesando: {mol2_path.name} + {itp_path.name}")

        for scheme, clf, reg in [
            ("AA", clf_aa, None),   # reg AA → TabTransformer
            ("UA", clf_ua, reg_ua), # reg UA → RF
        ]:
            try:
                result = validate(
                    mol2_path, itp_path, clf, reg,
                    artifacts, scheme, pipeline_dir, out_dir,
                    mol2_path.stem,
                    tab_model=tab_model if scheme == "AA" else None,
                    tab_scaler=tab_scaler if scheme == "AA" else None,
                    device=device,
                )
                all_results.append(result)
            except Exception as e:
                print(f"  [ERROR] {mol2_path.stem} {scheme}: {e}")
                import traceback; traceback.print_exc()

    summary_path = out_dir / "validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✔ Resumen guardado en: {summary_path}")

    section("RESUMEN VALIDACIÓN EXTERNA")
    print(f"{'Molécula':<12} {'Esquema':<8} {'Átomos':<8} {'Reg':<15} "
          f"{'Acc':<8} {'F1mac':<8} {'MAE(e)':<10} {'R²':<8}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['molecule']:<12} {r['scheme']:<8} {r['n_atoms']:<8} "
              f"{r['reg_model']:<15} "
              f"{r['classification']['accuracy']:<8.4f} "
              f"{r['classification']['f1_macro']:<8.4f} "
              f"{r['regression']['mae']:<10.4f} "
              f"{r['regression']['r2']:<8.4f}")


if __name__ == "__main__":
    main()