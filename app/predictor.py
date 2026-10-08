"""""
------------
Lógica de predicción: toma un .mol2, extrae features, aplica
preprocesamiento y predice atomtype y charge con los modelos AA y UA.
 
Modelos utilizados:
  - AA clasificación (atomtype): Random Forest E3 (clf_AA.joblib)
  - AA regresión (charge):       TabTransformer  (reg_AA.pt)
  - UA clasificación (atomtype): Random Forest E3 (clf_UA.joblib)
  - UA regresión (charge):       Random Forest E3 (reg_UA.joblib)
"""
 
import json
import math
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import load
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler
 
# =============================================================================
# IMPORTAR FUNCIONES DE EXTRACCIÓN DESDE EL PIPELINE
# =============================================================================
 
PIPELINE_DIR = Path("/home/marazi/proyectoParam/scripts/prueba2/Tesis_FINAL/version2")
sys.path.append(str(PIPELINE_DIR))
 
from extract_dataset_v3 import (
    parse_mol2,
    build_rdkit_mol,
    bond_length,
    angle_deg,
    is_planar_geometry,
    ATOMIC_MASSES,
    PLANAR_TYPES,
    ELECTRONEGATIVE,
    ELEMENT_NORMALIZE,
    BOND_TYPE_MAP,
)
 
# =============================================================================
# CONFIG — rutas a modelos y artifacts
# =============================================================================
 
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
 
CLF_AA    = MODELS_DIR / "clf_AA.joblib"   # RF E3 clasificación AA
REG_AA_PT = MODELS_DIR / "reg_AA.pt"       # TabTransformer regresión AA
CLF_UA    = MODELS_DIR / "clf_UA.joblib"   # RF E3 clasificación UA
REG_UA    = MODELS_DIR / "reg_UA.joblib"   # RF E3 regresión UA
ARTIFACTS = MODELS_DIR / "preprocessing_artifacts.json"
 
HASH_N_FEATURES = 32
COLS_TO_DROP    = ["degree_of_unsat", "n_dihedrals", "bonds_single", "tripos_type"]
BOOL_COLS       = ["is_planar", "is_in_ring", "is_aromatic"]
ID_COLS         = ["molecule", "atom_id", "atom_name", "bonded_to_element"]
TARGET_COLS     = ["atomtype", "charge"]
 
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
TABTRANSFORMER_COLS = NUMERIC_COLS + ELEMENT_COLS + NEIGHBOR_COLS + NEIGHBOR2_COLS  # 89
 
 
# =============================================================================
# TABTRANSFORMER — arquitectura (debe coincidir con el entrenamiento)
# =============================================================================
 
class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_features)
        ])
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
 
    def forward(self, x: torch.Tensor):
        tokens  = self.tokenizer(x)
        encoded = self.transformer(tokens)
        flat    = encoded.reshape(encoded.size(0), -1)
        logits  = self.head_cat(flat)
        charge  = self.head_reg(flat).squeeze(-1)
        return logits, charge
 
 
def load_tabtransformer(pt_path: Path, device: torch.device):
    """Carga el TabTransformer desde el checkpoint .pt."""
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    args        = ckpt["args"]
    n_classes   = len(ckpt["label_encoder_classes"])
    n_features  = len(TABTRANSFORMER_COLS)
 
    model = TabTransformerModel(
        n_features     = n_features,
        n_classes      = n_classes,
        d_model        = args.get("d_model", 64),
        nhead          = args.get("nhead", 8),
        num_layers     = args.get("num_layers", 3),
        dim_feedforward= args.get("d_model", 64) * 4,
        dropout        = args.get("dropout", 0.1),
    ).to(device)
 
    model.load_state_dict(ckpt["model_state"])
    model.eval()
 
    # Reconstruir scaler desde los parámetros guardados
    scaler        = StandardScaler()
    scaler.mean_  = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.n_features_in_ = n_features
 
    return model, scaler
 
 
def predict_charge_tabtransformer(model, scaler, df_proc, device):
    """Predice charge usando el TabTransformer."""
    # Asegurar que las columnas estén en el orden correcto
    for col in TABTRANSFORMER_COLS:
        if col not in df_proc.columns:
            df_proc[col] = 0
    X = df_proc[TABTRANSFORMER_COLS].values.astype(np.float32)
    X = scaler.transform(X)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, charge_pred = model(X_tensor)
    return charge_pred.cpu().numpy()
 
 
# =============================================================================
# EXTRACCIÓN DE FEATURES
# =============================================================================
 
def extract_features(mol2_path):
    atoms, bonds = parse_mol2(Path(mol2_path))
 
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
            if a == a2:
                continue
            for (d, _) in graph[a2]:
                if d == a1:
                    continue
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
        neighbor_config = "-".join(sorted(neighbor_info))
        avg_bond_len    = float(np.mean(bond_lengths)) if bond_lengths else 0.0
 
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
            "molecule":                atom["atom_name"],
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
 
def preprocess_for_prediction(df, artifacts, scheme):
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
# PREDICCIÓN PRINCIPAL
# =============================================================================
 
def predict_molecule(mol2_path, original_name=None):
    # Cargar modelos RF
    clf_aa = load(CLF_AA)
    clf_ua = load(CLF_UA)
    reg_ua = load(REG_UA)
 
    # Cargar TabTransformer para regresión AA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reg_aa_model, reg_aa_scaler = load_tabtransformer(REG_AA_PT, device)
 
    with open(ARTIFACTS) as f:
        artifacts = json.load(f)
 
    mol_name = Path(original_name).stem if original_name else Path(mol2_path).stem
 
    df_raw = extract_features(mol2_path)
    df_raw["molecule"] = mol_name
 
    df_aa_raw, df_ua_raw = split_AA_UA(df_raw)
 
    results = {}
 
    # ── AA ──────────────────────────────────────────────────────────────
    id_cols_aa = df_aa_raw[["molecule", "atom_id", "atom_name", "element"]].copy()
    df_aa_proc = preprocess_for_prediction(df_aa_raw, artifacts, "AA")
 
    # Clasificación AA → RF E3
    expected_cols_clf = list(clf_aa.feature_names_in_) if hasattr(clf_aa, "feature_names_in_") else []
    if expected_cols_clf:
        for col in expected_cols_clf:
            if col not in df_aa_proc.columns:
                df_aa_proc[col] = 0
        df_aa_proc_clf = df_aa_proc[expected_cols_clf]
    else:
        df_aa_proc_clf = df_aa_proc
 
    atomtypes_aa = clf_aa.predict(df_aa_proc_clf)
 
    # Regresión AA → TabTransformer
    charges_aa = predict_charge_tabtransformer(reg_aa_model, reg_aa_scaler, df_aa_proc.copy(), device)
 
    result_aa = id_cols_aa.copy()
    result_aa["atomtype_predicho"] = atomtypes_aa
    result_aa["charge_predicha"]   = [round(float(c), 4) for c in charges_aa]
    results["AA"] = result_aa.to_dict(orient="records")
 
    # ── UA ──────────────────────────────────────────────────────────────
    id_cols_ua = df_ua_raw[["molecule", "atom_id", "atom_name", "element"]].copy()
    df_ua_proc = preprocess_for_prediction(df_ua_raw, artifacts, "UA")
 
    expected_cols_ua = list(clf_ua.feature_names_in_) if hasattr(clf_ua, "feature_names_in_") else []
    if expected_cols_ua:
        for col in expected_cols_ua:
            if col not in df_ua_proc.columns:
                df_ua_proc[col] = 0
        df_ua_proc_aligned = df_ua_proc[expected_cols_ua]
    else:
        df_ua_proc_aligned = df_ua_proc
 
    atomtypes_ua = clf_ua.predict(df_ua_proc_aligned)
    charges_ua   = reg_ua.predict(df_ua_proc_aligned)
 
    result_ua = id_cols_ua.copy()
    result_ua["atomtype_predicho"] = atomtypes_ua
    result_ua["charge_predicha"]   = [round(float(c), 4) for c in charges_ua]
    results["UA"] = result_ua.to_dict(orient="records")
 
    return {
        "molecule": mol_name,
        "AA":       results["AA"],
        "UA":       results["UA"],
    }